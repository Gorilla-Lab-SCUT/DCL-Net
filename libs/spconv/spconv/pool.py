# Copyright 2019 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import time

import numpy as np
import spconv
import spconv.functional as Fsp
import torch
from spconv import ops
from spconv.modules import SparseModule
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class SparseMaxPool(SparseModule):
    def __init__(self,
                 ndim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 subm=False):
        super(SparseMaxPool, self).__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim

        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.subm = subm
        self.dilation = dilation

    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            out_spatial_shape = ops.get_conv_output_size(
                spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation)
        else:
            out_spatial_shape = spatial_shape
        outids, indice_pairs, indice_pairs_num = ops.get_indice_pairs(
            indices, batch_size, spatial_shape, self.kernel_size,
            self.stride, self.padding, self.dilation, 0, self.subm)
        
        out_features = Fsp.indice_maxpool(features, indice_pairs.to(device),
                                        indice_pairs_num.to(device), outids.shape[0])
        out_tensor = spconv.SparseConvTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class SparseMaxPool2d(SparseMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(SparseMaxPool2d, self).__init__(
            2,
            kernel_size,
            stride,
            padding,
            dilation)


class SparseMaxPool3d(SparseMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(SparseMaxPool3d, self).__init__(
            3,
            kernel_size,
            stride,
            padding,
            dilation)

class SparseFieldMaxPool(SparseModule):
    def __init__(self,
                 ndim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 subm=False):
        super(SparseFieldMaxPool, self).__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim

        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.subm = subm
        self.dilation = dilation
    def get_field_norm(self, features, field_list):
        start = 0
        feature_norm = []
        for l, m in enumerate(field_list):
            dim_field = 2*l+1
            for xx in range(m):
                feature_field      = features[:, start: start+dim_field]  # N dim_field
                start += dim_field
                feature_field_norm = feature_field.norm(dim = 1, keepdim = True)  # N 1
                feature_field_norm = feature_field_norm.repeat(1, dim_field)  # N dim_field
                feature_norm.append(feature_field_norm)
        feature_norm = torch.cat(feature_norm, dim = 1)  # N Σ(dim_field)
        return feature_norm
    def forward(self, input, field_list):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        feature_norms = self.get_field_norm(features, field_list)
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            out_spatial_shape = ops.get_conv_output_size(
                spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation)
        else:
            out_spatial_shape = spatial_shape
        outids, indice_pairs, indice_pairs_num = ops.get_indice_pairs(
            indices, batch_size, spatial_shape, self.kernel_size,
            self.stride, self.padding, self.dilation, 0, self.subm)
        
        out_features = Fsp.indice_fieldmaxpool(features, indice_pairs.to(device),
                                        indice_pairs_num.to(device), outids.shape[0], feature_norms)
        out_tensor = spconv.SparseConvTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class SparseFieldMaxPool2d(SparseFieldMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(SparseFieldMaxPool2d, self).__init__(
            2,
            kernel_size,
            stride,
            padding,
            dilation)


class SparseFieldMaxPool3d(SparseFieldMaxPool):
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(SparseFieldMaxPool3d, self).__init__(
            3,
            kernel_size,
            stride,
            padding,
            dilation)


class SparseAvgPool(SparseModule):
    def __init__(self,
                 ndim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 subm=False,
                 use_gs=True):
        super(SparseAvgPool, self).__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim

        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.subm = subm
        self.dilation = dilation
        self.use_gs   = use_gs

    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            out_spatial_shape = ops.get_conv_output_size(
                spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation)
        else:
            out_spatial_shape = spatial_shape
        outids, indice_pairs, indice_pairs_num = ops.get_indice_pairs(
            indices, batch_size, spatial_shape, self.kernel_size,
            self.stride, self.padding, self.dilation, 0, self.subm)
        
        out_features = Fsp.indice_avgpool(features, indice_pairs.to(device),
                                        indice_pairs_num.to(device), outids.shape[0], self.use_gs)
        out_tensor = spconv.SparseConvTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class SparseAvgPool2d(SparseAvgPool):
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 use_gs=True):
        super(SparseAvgPool2d, self).__init__(
            2,
            kernel_size,
            stride,
            padding,
            dilation,
            use_gs = use_gs)


class SparseAvgPool3d(SparseAvgPool):
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 use_gs=True):
        super(SparseAvgPool3d, self).__init__(
            3,
            kernel_size,
            stride,
            padding,
            dilation,
            use_gs = use_gs)
