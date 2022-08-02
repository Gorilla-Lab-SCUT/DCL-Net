import torch.nn as nn
import torch
from libs.pointnet_lib.pointnet2_utils import knn
from utils.transform3D import normalize_vector
class Head_MultiLayerPerceptron(nn.Module):
    def __init__(self, list_dim, list_act, list_bn, list_drop) -> None:
        super().__init__()
        self.layers = []
        dim_inp = list_dim[0]
        for dim, act, bn, drop in zip(list_dim[1:], list_act, list_bn, list_drop):
            self.layers.append(nn.Conv1d(dim_inp, dim, 1))

            if act == "relu":
                self.layers.append(nn.ReLU())
            elif act == "sigmoid":
                self.layers.append(nn.Sigmoid())
            elif act == "tanh":
                self.layers.append(nn.tanh())
            elif act == "none":
                pass
            else:
                raise NotImplementedError
            
            if bn:
                self.layers.append(nn.BatchNorm1d(dim))
            
            if drop>0.0:
                self.layers.append(nn.Dropout(drop))
            dim_inp = dim
        self.layers = nn.Sequential(*self.layers)
    def forward(self, input):
        # input/output : B C N
        output = self.layers(input)
        return output
def ortho9d2matrix(x_raw, y_raw, z_raw):
    '''
    Description: get the rotation matrix computed by the two orthored vectors
    
    Args:
    
    '''
    x = normalize_vector(x_raw, 'torch')
    y = normalize_vector(y_raw, 'torch')
    z = normalize_vector(z_raw, 'torch')
    
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)
    matrix = torch.cat((x,y,z), dim=2)
    bs     = matrix.shape[0]
    U, S, V = torch.svd(matrix)
    sigma   = torch.ones([bs, 3]).cuda()
    sigma[:, -1] = torch.bmm(U, V.transpose(1,2)).det()
    sigma        = torch.diag_embed(sigma)
    matrix       = U @ sigma @ V.transpose(1,2)
    return matrix
class Refiner(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.MLP_share = Head_MultiLayerPerceptron(
            [256 + 3    ,  512   ,   512,   1024   ],
            [     "relu" , "relu", "relu" ],
            [      False ,  False,  False ],
            [      0.0   ,  0.0  ,  0.0   ],
        )
        self.regressor_rot2          = Head_MultiLayerPerceptron(
            [1024,  512,  128,   9  ],
            [     "relu", "relu", "none"],
            [      False,  False,  False],
            [      0.0  ,  0.0  ,  0.0  ],
        )
        self.regressor_trans2          = Head_MultiLayerPerceptron(
            [1024,  512,   128,   3 ],
            [     "relu", "relu", "none"],
            [      False,  False,  False],
            [      0.0  ,  0.0  ,  0.0  ],
        )
    def forward(self, input_dict):
        input_features = input_dict["input_features"]
        conf           = input_dict["conf"]
        conf_softmax   = torch.softmax(conf.unsqueeze(1), dim=2)[:,:,:1024]
        shared_feature = self.MLP_share(input_features)
        shared_feature = (shared_feature * conf_softmax).sum(dim=2, keepdim=True)
        
        ortho9d_pred2   = self.regressor_rot2(shared_feature).squeeze(-1)
        delta_t         = self.regressor_trans2(shared_feature).squeeze(-1)
        rot_x_pred2   = ortho9d_pred2[:, :3]
        rot_y_pred2   = ortho9d_pred2[:, 3:6]
        rot_z_pred2   = ortho9d_pred2[:, 6:]
        delta_R       = ortho9d2matrix(rot_x_pred2, rot_y_pred2, rot_z_pred2)
        prediction = {
                "trans_pred": delta_t,
                "rot_pred"  : delta_R,
            }
        return prediction
    
    

class losses_refiner(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
    def forward(self, loss_inp_pred_refiner, trans_cur, rot_cur, points_tmp, sym_flag, loss_inp_gt):
        rot_pred         = rot_cur
        trans_pred       = trans_cur
        delta_rot_pred   = loss_inp_pred_refiner["rot_pred"]
        delta_trans_pred = loss_inp_pred_refiner["trans_pred"]

        rot_gt     = loss_inp_gt["rot_gt"].cuda()
        trans_gt   = loss_inp_gt["trans_gt"].cuda()


        points_tmp_posed_pred = torch.bmm(points_tmp, delta_rot_pred.transpose(1,2)) + delta_trans_pred.unsqueeze(1)
        points_tmp_posed_gt   = torch.bmm(points_tmp, rot_gt.transpose(1,2))   + trans_gt.unsqueeze(1)
        
        points_tmp_posed_refined = torch.bmm(points_tmp_posed_pred, rot_pred.transpose(1,2)) + trans_pred.unsqueeze(1)

        loss_pose = ((1-sym_flag).unsqueeze(1) * self.L2_Dis(points_tmp_posed_refined, points_tmp_posed_gt) + sym_flag.unsqueeze(1) * self.CD_Dis(points_tmp_posed_refined, points_tmp_posed_gt)).mean(dim = 1).mean()
        
        loss_all  = loss_pose 
        
        losses = {
            "loss_pose": loss_pose,
            "loss_all" : loss_all
        }
        return losses
    def L2_Dis(self, pred, target):
        return torch.norm(pred - target, dim=2)

    def CD_Dis(self, pred, target):
        dis = torch.norm(pred.unsqueeze(2) - target.unsqueeze(1), dim=3)
        dis1 = torch.min(dis, 2)[0]
        dis2 = torch.min(dis, 1)[0]
        return 0.5*(dis1+dis2)
    @staticmethod
    def get_cano_label(points_tmp, points_inp, rot_pred, trans_gt):
        points_inp_cano = torch.bmm((points_inp - trans_gt), rot_pred)
        _, idx          = knn(1, points_inp_cano, points_tmp)
        label_cano      = torch.gather(points_tmp, 1, idx.long().repeat(1,1,3))
        return label_cano