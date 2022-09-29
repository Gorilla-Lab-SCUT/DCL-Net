import ipdb
import numpy as np
import sys
import spconv
import torch
import torch.nn as nn
from models.Modules import Aligner, Backbone_SPCONV, BasicBlock_3DCONV, Ops_GetPointFeat_spconv, Head_MultiLayerPerceptron
from functools import partial
import torch.nn as nn
from libs.pointnet_lib.pointnet2_utils import knn

from libs.pointgroup_ops.functions import pointgroup_ops
from utils.transform3D import normalize_vector

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

class Network(nn.Module):
    def __init__(self, cfg, mode="train") -> None:
        super().__init__()
        self.voxelization_mode = cfg.voxelization_mode
        self.unit_voxel_extent = np.array(cfg.unit_voxel_extent)
        self.mode = mode
        self.n_inp = cfg.n_inp
        self.n_tmp = cfg.n_tmp

        self.backbone_dims = [
            7, 16, 32, 32, 64, 64, 128, 128, 256
        ]
        self.backbone_stride_layers = [1, 3, 5]
        self.backbone_inp = Backbone_SPCONV(dims=self.backbone_dims, stride_layers=self.backbone_stride_layers, cfg=cfg.backbone)
        self.backbone_tmp = Backbone_SPCONV(dims=self.backbone_dims, stride_layers=self.backbone_stride_layers, cfg=cfg.backbone)

        self.stage1_get_point_feats = Ops_GetPointFeat_spconv(scale_lists=[2, 4, 6, 8], unit_voxel_extent=self.unit_voxel_extent, voxel_num_limit=[64, 64, 64])

        common_param_decouple = {
                "size"   : 1,
                "bias"   : False,
                "stride" : 1,
                "padding": 0,
                "norm"   : True,
                "act"    : "relu",
                "drop"   : 0.0

        }
        Basiclock_1x1 = partial(
            BasicBlock_3DCONV, **common_param_decouple
        )
        self.disengage_Xc_p1 = nn.Sequential(*[
            Basiclock_1x1(dim_in = 480, dim_out = 256),
            Basiclock_1x1(dim_in = 256, dim_out = 256) 
        ])
        self.disengage_Xc_m1 = nn.Sequential(*[
            Basiclock_1x1(dim_in = 480, dim_out = 256),
            Basiclock_1x1(dim_in = 256, dim_out = 64) ,
        ])
        self.disengage_Yo_p1 = nn.Sequential(*[
            Basiclock_1x1(dim_in = 480, dim_out = 256),
            Basiclock_1x1(dim_in = 256, dim_out = 256) 
        ])
        self.disengage_Yo_m1 = nn.Sequential(*[
            Basiclock_1x1(dim_in = 480, dim_out = 256),
            Basiclock_1x1(dim_in = 256, dim_out = 64) ,
        ])
        
        self.disengage_Xc_p2 = nn.Sequential(*[
            Basiclock_1x1(dim_in = 480, dim_out = 256),
            Basiclock_1x1(dim_in = 256, dim_out = 256) 
        ])
        self.disengage_Xc_m2 = nn.Sequential(*[
            Basiclock_1x1(dim_in = 480, dim_out = 256),
            Basiclock_1x1(dim_in = 256, dim_out = 64) ,
        ])
        self.disengage_Yo_p2 = nn.Sequential(*[
            Basiclock_1x1(dim_in = 480, dim_out = 256),
            Basiclock_1x1(dim_in = 256, dim_out = 256)
        ])
        self.disengage_Yo_m2 = nn.Sequential(*[
            Basiclock_1x1(dim_in = 480, dim_out = 256),
            Basiclock_1x1(dim_in = 256, dim_out = 64) ,
        ])
        self.neck_cross_att         = Aligner()
        self.regressor_Xo          = Head_MultiLayerPerceptron(
            [  256,  256  ,  128  ,  3    ],
            [     "relu", "relu", "none"],
            [      False,  False,  False],
            [      0.0  ,  0.0  ,  0.0  ],
        )
        self.regressor_Yc            = Head_MultiLayerPerceptron(
            [  256,  256  ,  128  ,  3    ],
            [     "relu", "relu", "none"],
            [      False,  False,  False],
            [      0.0  ,  0.0  ,  0.0  ],
        )
        self.regressor_conf          = Head_MultiLayerPerceptron(
            [64*2,  128,  128,   1  ],
            [     "relu", "relu", "none"],
            [      False,  False,  False],
            [      0.0  ,  0.0  ,  0.0  ],
        )
        self.regressor_conf_bi          = Head_MultiLayerPerceptron(
            [64*2,  128,  128,   1  ],
            [     "relu", "relu", "none"],
            [      False,  False,  False],
            [      0.0  ,  0.0  ,  0.0  ],
        )
        self.neck_fuser                  = Head_MultiLayerPerceptron(
            [ 256*2,  512, 512, 1024],
            [       "relu", "relu", "relu"],
            [        True,  True,  True],
            [        0.0,     0.0,    0.0  ],
        )
        self.neck_fuser_bi               = Head_MultiLayerPerceptron(
            [ 256*2,  512, 512, 1024],
            [       "relu", "relu", "relu"],
            [        True,  True,  True],
            [        0.0,     0.0,    0.0  ],
        )

        self.regressor_rot          = Head_MultiLayerPerceptron(
            [1024,  512,  128,   9  ],
            [     "relu", "relu", "none"],
            [      False,  False,  False],
            [      0.0  ,  0.0  ,  0.0  ],
        )
        self.regressor_trans          = Head_MultiLayerPerceptron(
            [1024,  512,   128,   3 ],
            [     "relu", "relu", "none"],
            [      False,  False,  False],
            [      0.0  ,  0.0  ,  0.0  ],
        )



    def forward(self, data):
        # get data
        feats_inp    = data["inp"]['feats'].cuda()
        points_inp   = data["inp"]['feats'][:, 4:].cuda()
        RGB_inp      = data["inp"]['feats'][:, 1:4].cuda()
        v2p_maps_inp = data["inp"]['v2p_maps'].cuda()
        occupied_voxels_inp = data["inp"]['occupied_voxels'].cuda().int()

        feats_tmp    = data["tmp"]['feats'].cuda()
        points_tmp   = data["tmp"]['feats'][:, 4:].cuda()
        RGB_tmp      = data["tmp"]['feats'][:, 1:4].cuda()
        v2p_maps_tmp = data["tmp"]['v2p_maps'].cuda()
        occupied_voxels_tmp = data["tmp"]['occupied_voxels'].cuda().int()

        voxel_num_limit = data['voxel_num_limit'].numpy().astype(np.int)
        batch_offsets   = data['batch_offsets'].cuda()
        b = batch_offsets.size(0)-1

        # backbone
        feats_inp = pointgroup_ops.voxelization(feats_inp, v2p_maps_inp, self.voxelization_mode)
        feats_inp = spconv.SparseConvTensor(feats_inp, occupied_voxels_inp.int(), voxel_num_limit, b)
        feats1_inp, feats2_inp, feats3_inp, feats4_inp = self.backbone_inp(feats_inp)

        feats_tmp = pointgroup_ops.voxelization(feats_tmp, v2p_maps_tmp, self.voxelization_mode)
        feats_tmp = spconv.SparseConvTensor(feats_tmp, occupied_voxels_tmp.int(), voxel_num_limit, b)
        feats1_tmp, feats2_tmp, feats3_tmp, feats4_tmp = self.backbone_tmp(feats_tmp)
        
        points_inp = points_inp.view(b, self.n_inp, 3)
        points_inp = points_inp[:,:self.n_inp, :].reshape(-1, 3)
        batch_ids_inp = torch.arange(b).unsqueeze(1).repeat(1, points_inp.size(0)//b).cuda().view(-1, 1)
        F_Xc = self.stage1_get_point_feats(points_inp, batch_ids_inp, feats1_inp, feats2_inp, feats3_inp, feats4_inp)

        # bi-direction FDA
        F_Xc = F_Xc.view(b, self.n_inp, -1).transpose(1,2)[:,:,:, None, None]
        F_Xc_p1 = self.disengage_Xc_p1(F_Xc).squeeze(-1).squeeze(-1)
        F_Xc_m1 = self.disengage_Xc_m1(F_Xc).squeeze(-1).squeeze(-1)
        F_Xc_p2 = self.disengage_Xc_p2(F_Xc).squeeze(-1).squeeze(-1)
        F_Xc_m2 = self.disengage_Xc_m2(F_Xc).squeeze(-1).squeeze(-1)

        batch_ids_tmp   = torch.arange(b).unsqueeze(1).repeat(1, points_tmp.size(0)//b).cuda().view(-1, 1)
        F_Yo = self.stage1_get_point_feats(points_tmp, batch_ids_tmp, feats1_tmp, feats2_tmp, feats3_tmp, feats4_tmp)
        F_Yo = F_Yo.view(b, self.n_tmp, -1).transpose(1,2)[:,:,:, None, None]
        F_Yo_p1 = self.disengage_Yo_p1(F_Yo).squeeze(-1).squeeze(-1)
        F_Yo_m1 = self.disengage_Yo_m1(F_Yo).squeeze(-1).squeeze(-1)
        F_Yo_p2 = self.disengage_Yo_p2(F_Yo).squeeze(-1).squeeze(-1)
        F_Yo_m2 = self.disengage_Yo_m2(F_Yo).squeeze(-1).squeeze(-1)
        
        points_tmp = points_tmp.view(b, self.n_tmp, -1)
        points_inp = points_inp.view(b, self.n_inp, -1)
        RGB_tmp    = RGB_tmp.view(b, self.n_tmp, -1)
        RGB_inp    = RGB_inp.view(b, self.n_inp, -1)
        F_Xo_p, attention_map = self.neck_cross_att(F_Xc_m1, F_Yo_m1, F_Yo_p1)
        Xo_pred = self.regressor_Xo(F_Xo_p)
        
        F_Yc_p, attention_map_bi = self.neck_cross_att(F_Yo_m2, F_Xc_m2, F_Xc_p2)
        Yc_pred = self.regressor_Yc(F_Yc_p)

        # confidence
        F_Xo_m = torch.bmm(F_Yo_m1, attention_map)
        F_m1 = torch.cat([F_Xc_m1, F_Xo_m], dim=1)
        F_Yc_m = torch.bmm(F_Xc_m2, attention_map_bi)
        F_m2 = torch.cat([F_Yc_m, F_Yo_m2], dim=1)
        conf_1 = self.regressor_conf(F_m1)
        conf_2 = self.regressor_conf_bi(F_m2)
        conf = torch.sigmoid(torch.cat([conf_1, conf_2], dim=2))
        conf_softmax = torch.softmax(conf, dim=2)

        # head
        F_p1 = torch.cat([F_Xc_p1, F_Xo_p], dim = 1)
        F_p2 = torch.cat([F_Yc_p, F_Yo_p2], dim = 1)
        F_p1 = self.neck_fuser(F_p1)
        F_p2 = self.neck_fuser_bi(F_p2)
        F_p = torch.cat([F_p1, F_p2], dim = 2)
        F_p_wei = torch.sum(F_p * conf_softmax, dim=2, keepdims=True)

        ortho9d_pred = self.regressor_rot(F_p_wei).squeeze(-1)
        rot_x_pred   = ortho9d_pred[:, :3]
        rot_y_pred   = ortho9d_pred[:, 3:6]
        rot_z_pred   = ortho9d_pred[:, 6:]
        rot_pred     = ortho9d2matrix(rot_x_pred, rot_y_pred, rot_z_pred)
        trans_pred   = self.regressor_trans(F_p_wei).squeeze(-1)

        
        if self.mode == 'test':
            prediction = {
                    "trans_pred": trans_pred,
                    "rot_pred"  : rot_pred  ,
                    'conf': conf.squeeze(1),
                    "F_Xo_p": F_Xo_p,
            }

        else:
            prediction = {
                    "trans_pred": trans_pred,
                    "rot_pred"  : rot_pred  ,
                    'sym_flag'  : data['flags'].cuda(),
                    'conf'      : conf.squeeze(1),
                    "Xo_pred"   : Xo_pred.transpose(1,2),
                    "Yc_pred"   : Yc_pred.transpose(1,2),
                    "F_Xo_p"    : F_Xo_p,
                }

        data["labels"]["points_tmp"] = points_tmp
        data["labels"]["points_inp"] = points_inp
        return prediction

class losses(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
    def forward(self, loss_inp_pred, loss_inp_gt):
        rot_pred   = loss_inp_pred["rot_pred"]
        trans_pred = loss_inp_pred["trans_pred"]
        sym_flag   = loss_inp_pred["sym_flag"]

        rot_gt     = loss_inp_gt["rot_gt"].cuda()
        trans_gt   = loss_inp_gt["trans_gt"].cuda()

        points_tmp = loss_inp_gt["points_tmp"]
        points_inp = loss_inp_gt["points_inp"]

        conf = loss_inp_pred['conf']

        points_tmp_posed_pred = torch.bmm(points_tmp, rot_pred.transpose(1,2)) + trans_pred.unsqueeze(1)
        points_tmp_posed_gt   = torch.bmm(points_tmp, rot_gt.transpose(1,2))   + trans_gt.unsqueeze(1)

        loss_pose = ((1-sym_flag).unsqueeze(1) * self.L2_Dis(points_tmp_posed_pred, points_tmp_posed_gt) + sym_flag.unsqueeze(1) * self.CD_Dis(points_tmp_posed_pred, points_tmp_posed_gt)).mean(dim = 1).mean()

        Xo_pred  = loss_inp_pred["Xo_pred"]
        Yc_pred    = loss_inp_pred["Yc_pred"]
        points_inp_posed_pred = (torch.bmm(points_inp-trans_pred.unsqueeze(1), rot_pred)).detach()
        points_inp_posed_gt = torch.bmm(points_inp-trans_gt.unsqueeze(1), rot_gt).detach()
        loss_Xo = (1-sym_flag).unsqueeze(1) * self.L2_Dis(Xo_pred, points_inp_posed_gt) + 0.5 * sym_flag.unsqueeze(1) * (self.CD_Dis(Xo_pred, points_tmp)+ self.L2_Dis(Xo_pred, points_inp_posed_pred))
        loss_Xo_ = loss_Xo.mean()
        
        loss_Yc = (1-sym_flag).unsqueeze(1) * self.L2_Dis(Yc_pred, points_tmp_posed_gt) + 0.5 * sym_flag.unsqueeze(1) * (self.CD_Dis(Yc_pred, points_tmp_posed_gt)+ self.L2_Dis(Yc_pred, points_tmp_posed_pred.detach()))
        loss_Yc_ = loss_Yc.mean()
        loss_conf = torch.mean(torch.cat([loss_Xo, loss_Yc], dim = 1).detach()*conf - 0.01 * torch.log(conf))

        loss_all  = loss_pose + 5*loss_Xo_ + 1*loss_Yc_ + 1*loss_conf

        losses = {
            "loss_pose": loss_pose,
            "loss_Xo"  : loss_Xo_,
            "loss_Yc"  : loss_Yc_,
            "loss_conf": loss_conf,
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