
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

import spconv.ops as ops
import torch
from torch import nn
from torch.autograd import Function


class SparseConvFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out):
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            filters)
        return ops.indice_conv(features, filters, indice_pairs, indice_pair_num, num_activate_out, False)

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(features, filters, grad_output.contiguous(), indice_pairs, indice_pair_num, False)

        return input_bp, filters_bp, None, None, None

class SparseInverseConvFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out):
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            filters)
        return ops.indice_conv(features, filters, indice_pairs, indice_pair_num, num_activate_out, True, False)

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(features, filters, grad_output.contiguous(), indice_pairs, indice_pair_num, True, False)

        return input_bp, filters_bp, None, None, None


class SubMConvFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out):
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            filters)
        return ops.indice_conv(features, filters, indice_pairs, indice_pair_num, num_activate_out, False, True)

    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(features, filters, grad_output.contiguous(), indice_pairs, indice_pair_num, False, True)

        return input_bp, filters_bp, None, None, None


class SparseMaxPoolFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            indice_pairs,
            indice_pair_num,
            num_activate_out):
        out = ops.indice_maxpool(features, indice_pairs, indice_pair_num, num_activate_out)
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            out)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out = ctx.saved_tensors
        input_bp = ops.indice_maxpool_backward(features, out, grad_output.contiguous(), indice_pairs, indice_pair_num)
        return input_bp, None, None, None

class SparseFieldMaxPoolFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            feature_norms):
        out = ops.indice_fieldmaxpool(features, indice_pairs, indice_pair_num, num_activate_out, feature_norms)
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            out)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out = ctx.saved_tensors
        input_bp = ops.indice_maxpool_backward(features, out, grad_output.contiguous(), indice_pairs, indice_pair_num)
        return input_bp, None, None, None, None


class SparseAvgPoolFunction(Function):
    @staticmethod
    def forward(
            ctx,
            features,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            use_gs=True):
        if not use_gs:
            summaryrf = ops.get_indice_summaryrf(indice_pairs, indice_pair_num, num_activate_out)
            # print(summaryrf.shape, "  ", summaryrf.dtype)
        else:
            kernel_volume = indice_pairs.shape[0]
            # print(kernel_volume)
            summaryrf = indice_pairs.new_zeros(num_activate_out) + int(kernel_volume)
            # print(summaryrf.shape, "  ", summaryrf.dtype)
            # print(features.shape, "  ", features.dtype)
        out = ops.indice_avgpool(features, indice_pairs, indice_pair_num, num_activate_out, summaryrf)
        ctx.save_for_backward(
            indice_pairs,
            indice_pair_num,
            features,
            out,
            summaryrf)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out, summaryrf = ctx.saved_tensors
        input_bp = ops.indice_avgpool_backward(features, out, grad_output.contiguous(), indice_pairs, indice_pair_num, summaryrf)
        return input_bp, None, None, None, None

indice_conv = SparseConvFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_maxpool = SparseMaxPoolFunction.apply
indice_avgpool = SparseAvgPoolFunction.apply
indice_fieldmaxpool = SparseFieldMaxPoolFunction.apply