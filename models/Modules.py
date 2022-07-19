
from functools import partial
import torch
import spconv
import numpy as np
import torch.nn as nn
from spconv import SparseAvgPool3d

from libs.pointnet_sp import pointnet2_utils as pointnet2_utils_sp

#! -----------------------------------BASICBLOCK-----------------------------------
class BasicBlock_SPCONV(nn.Module):
    def __init__(self, subm, dim_in, dim_out, bias, size, stride, padding, norm, act, drop, indice_key) -> None:
        super().__init__()
        self.dim_in  = dim_in
        self.dim_out = dim_out
        self.size    = size
        self.stride  = stride
        self.padding = padding
        self.norm    = norm
        self.act     = act
        self.drop    = drop
        self.indice_key = indice_key
        self.subm    = subm
        self.bias    = bias
        self.layers   = []
        if not self.subm:
            self.layers.append(
                spconv.SparseConv3d(self.dim_in, self.dim_out, self.size, (self.stride, ) * 3, padding=self.padding, bias=self.bias, indice_key=self.indice_key),
            )
        else:
            self.layers.append(
                spconv.SubMConv3d(  self.dim_in, self.dim_out, self.size                     , padding=self.padding, bias=self.bias, indice_key=self.indice_key),
            )
        
        if self.norm:
            self.layers.append(nn.BatchNorm1d(self.dim_out))
        
        if  self.act == "relu":
            self.layers.append(nn.ReLU())
        elif self.act == "sigmoid":
            self.layers.append(nn.Sigmoid())
        elif self.act == "tanh":
            self.layers.append(nn.tanh())
        elif self.act == "none":
            pass
        else:
            raise NotImplementedError
        

        if self.drop > 0:
            self.layers.append(nn.Dropout(self.drop))

        self.layers = spconv.SparseSequential(*self.layers)
    def forward(self, input):
        output = self.layers(input)
        return output
class BasicBlock_3DCONV(nn.Module):
    def __init__(self, dim_in, dim_out, bias, size, stride, padding, norm, act, drop) -> None:
        super().__init__()
        self.dim_in  = dim_in
        self.dim_out = dim_out
        self.size    = size
        self.stride  = stride
        self.padding = padding
        self.norm    = norm
        self.act     = act
        self.drop    = drop
        self.bias    = bias
        self.layers   = []

        self.layers.append(nn.Conv3d(
            dim_in, dim_out, size, stride, padding, bias = bias
        ))

        if self.norm:
            self.layers.append(nn.BatchNorm3d(self.dim_out))
        
        if  self.act == "relu":
            self.layers.append(nn.ReLU())
        elif self.act == "sigmoid":
            self.layers.append(nn.Sigmoid())
        elif self.act == "tanh":
            self.layers.append(nn.tanh())
        elif self.act == "none":
            pass
        else:
            raise NotImplementedError
        

        if self.drop > 0:
            self.layers.append(nn.Dropout(self.drop))

        self.layers = nn.Sequential(*self.layers)
    def forward(self, input):
        output = self.layers(input)
        return output
#! ------------------------------------BACKBONE------------------------------------

class Backbone_SPCONV(nn.Module):
    def __init__(self, dims, stride_layers, cfg, norm=True):
        super().__init__()
        self.downsample_by_pooling = cfg.downsample_by_pooling
        self.dims = dims
        self.stride_layers = stride_layers
        self.N_layser_backbone = len(self.dims)
        self.common_params = {
            "bias": False,
            "act" : "relu",
            "drop": 0.0,
            "norm": norm,
        }
        basic_block = partial(BasicBlock_SPCONV, **self.common_params)

        modules = [[] for i in range(len(self.stride_layers)+1)]
        module_index = 0
        for i in range(self.N_layser_backbone-1):
            dim_in = self.dims[i]
            dim_out= self.dims[i+1]
            if i in self.stride_layers:
                stride      = 2
            else:
                stride      = 1

            if (i-1) in self.stride_layers or i==0:
                subm = False
                indice_key = "spconv_" + str(module_index)
            else:
                subm = True
                indice_key = "subm_spconv_" + str(module_index)

            modules[module_index].append(
                basic_block(
                            subm    = subm,
                            dim_in  = dim_in, 
                            dim_out = dim_out,
                            size    = cfg.kernel_size, 
                            stride  = 1,
                            padding = cfg.kernel_size//2,
                            indice_key = indice_key
                )
            )
            if stride>1:
                module_index += 1
            # print("subm:{0}, dim_in:{1}, dim_out:{2}, size:{3}, stride_conv:{4}".format(subm, dim_in, dim_out, cfg.kernel_size, 1))
        
        self.module1 = nn.Sequential(*modules[0])
        self.module2 = nn.Sequential(*modules[1])
        self.module3 = nn.Sequential(*modules[2])
        self.module4 = nn.Sequential(*modules[3])
        self.pool = SparseAvgPool3d(kernel_size=cfg.kernel_size, stride=2, padding = cfg.kernel_size//2, use_gs=False)

    def forward(self, inputs):
        feats1 = self.pool(self.module1(inputs))
        feats2 = self.pool(self.module2(feats1))
        feats3 = self.pool(self.module3(feats2))
        feats4 = self.pool(self.module4(feats3))

        return feats1, feats2, feats3, feats4

#! --------------------------------------NECK--------------------------------------
class Aligner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, RI_1, RI_2, RE_2):
        attention_map = self.softmax(torch.bmm(RI_2.transpose(1,2), RI_1))
        RE_embed    = torch.bmm(RE_2, attention_map)
        return RE_embed, attention_map


#! --------------------------------------HEAD--------------------------------------
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
        output = self.layers(input)
        return output

#! ------------------------------------OPERATION------------------------------------
def Ops_tensor2points(tensor, offset=(0., -40., -3.), voxel_extent=(.1, .1, .2)):
    # tensor: sparse tensor
    indices = tensor.indices.float()
    offset = torch.Tensor(offset).to(indices.device)
    voxel_extent = torch.Tensor(voxel_extent).to(indices.device)
    indices[:, 1:] = indices[:, 1:] * voxel_extent + offset + .5 * voxel_extent

    return tensor.features, indices

def Ops_nearest_neighbor_interpolate(target_points, query_points, query_feats):
    """
    :param target_points: (n, 4) tensor of the bxyz positions of the unknown features
    :param query_points: (m, 4) tensor of the bxyz positions of the known features
    :param query_feats: (m, C) tensor of features to be propigated
    :return:
        interpolated_feats: (n, C) tensor of the features of the unknown features
    """
    dist, idx = pointnet2_utils_sp.three_nn(target_points, query_points)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils_sp.three_interpolate(query_feats, idx, weight)

    return interpolated_feats
class Ops_GetPointFeat_spconv(nn.Module):
    def __init__(self, scale_lists=[2,4,8,16], unit_voxel_extent=np.array([0.015, 0.015, 0.015]), voxel_num_limit=np.array([64, 64, 64])):
        super().__init__()
        self.scale_lists = scale_lists
        self.unit_voxel_extent = unit_voxel_extent
        self.voxel_num_limit = voxel_num_limit
        self.offset = -0.5 * self.unit_voxel_extent * self.voxel_num_limit

    def forward(self, points, batch_ids, feats1, feats2, feats3, feats4):
        offset = self.offset
        scale_lists = self.scale_lists
        points = torch.cat([batch_ids.view(-1, 1).float(), points], 1)
        
        point_feats_all = []
        feat_lists = [feats1, feats2, feats3, feats4]

        for i in range(4):
            scale = scale_lists[i]
            feats = feat_lists[i]
            vx_feats, vx_points = Ops_tensor2points(feats, offset, self.unit_voxel_extent*scale)
            point_feats = Ops_nearest_neighbor_interpolate(points, vx_points, vx_feats)
            point_feats_all.append(point_feats)
        point_feats = torch.cat(point_feats_all, dim = 1)
        return point_feats
