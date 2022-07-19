// Copyright 2019 Yan Yan
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SPARSE_POOL_OP_H_
#define SPARSE_POOL_OP_H_

#include <cuda_runtime_api.h>
#include <spconv/maxpool.h>
#include <spconv/avgpool.h>
#include <spconv/summaryRF.h>
#include <torch/script.h>
#include <torch_utils.h>
#include <utility/timer.h>

namespace spconv {
template <typename T>
torch::Tensor indiceMaxPool(torch::Tensor features, torch::Tensor indicePairs,
                          torch::Tensor indiceNum, int64_t numAct) {
  auto device = features.device().type();
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numAct, numInPlanes}, options);
  // torch::Tensor smallest = torch::tensor({-1e50}, options);
  // output = output + smallest;
  double totalTime = 0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    if (device == torch::kCPU) {
      functor::SparseMaxPoolForwardFunctor<tv::CPU, T, int> forwardFtor;
      forwardFtor(tv::CPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot);
    } else {
      functor::SparseMaxPoolForwardFunctor<tv::GPU, T, int> forwardFtor;
      forwardFtor(tv::TorchGPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot);
      TV_CHECK_CUDA_ERR();
    }
    // totalTime += timer.report() / 1000.0;
  }
  // std::cout << "maxpool forward time " << totalTime << std::endl;
  return output;
}

template <typename T>
torch::Tensor indiceFieldMaxPool(torch::Tensor features, torch::Tensor indicePairs,
                          torch::Tensor indiceNum, int64_t numAct,
                          torch::Tensor feature_norms) {
  auto device = features.device().type();
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numAct, numInPlanes}, options);
  torch::Tensor output_norm = torch::zeros({numAct, numInPlanes}, options);
  torch::Tensor smallest = torch::tensor({-1e50}, options);
  output_norm = output_norm + smallest; 
  output = output + smallest;
  double totalTime = 0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    if (device == torch::kCPU) {
      functor::SparseFieldMaxPoolForwardFunctor<tv::CPU, T, int> forwardFtor;
      forwardFtor(tv::CPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot,
                  tv::torch2tv<const T>(feature_norms), tv::torch2tv<T>(output_norm));
    } else {
      functor::SparseFieldMaxPoolForwardFunctor<tv::GPU, T, int> forwardFtor;
      forwardFtor(tv::TorchGPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot,
                  tv::torch2tv<const T>(feature_norms), tv::torch2tv<T>(output_norm));
      TV_CHECK_CUDA_ERR();
    }
    // totalTime += timer.report() / 1000.0;
  }
  // std::cout << "maxpool forward time " << totalTime << std::endl;
  return output;
}

template <typename T>
torch::Tensor indiceMaxPoolBackward(torch::Tensor features,
                                  torch::Tensor outFeatures,
                                  torch::Tensor outGrad, torch::Tensor indicePairs,
                                  torch::Tensor indiceNum) {
  auto device = features.device().type();
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
    auto kernelVolume = indicePairs.size(0);
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    if (device == torch::kCPU) {
      functor::SparseMaxPoolBackwardFunctor<tv::CPU, T, int> backwardFtor;
      backwardFtor(tv::CPU(), tv::torch2tv<const T>(outFeatures),
                   tv::torch2tv<const T>(features),
                   tv::torch2tv<const T>(outGrad), tv::torch2tv<T>(inputGrad),
                   tv::torch2tv<const int>(indicePairs).subview(i), nHot);
    } else {
      functor::SparseMaxPoolBackwardFunctor<tv::GPU, T, int> backwardFtor;
      backwardFtor(tv::TorchGPU(), tv::torch2tv<const T>(outFeatures),
                   tv::torch2tv<const T>(features),
                   tv::torch2tv<const T>(outGrad), tv::torch2tv<T>(inputGrad),
                   tv::torch2tv<const int>(indicePairs).subview(i), nHot);
      TV_CHECK_CUDA_ERR();
    }
  }
  return inputGrad;
}

torch::Tensor indiceSummaryRF(torch::Tensor indicePairs,
                              torch::Tensor indiceNum, 
                              int64_t numAct){
  auto device             = indicePairs.device().type();  // 获取设备（判断是CPU还是GPU）
  auto kernelVolume       = indicePairs.size(0);
  auto indicePairNumCpu   = indiceNum.to({torch::kCPU});   // 将indice pair放到CPU中
  auto options            = torch::TensorOptions().dtype(indicePairs.dtype()).device(indicePairs.device());  // 获取 新构建的tensor的数据类型以及数据的设备
  torch::Tensor summarRFs = torch::zeros({numAct}, options);
  for(int i = 0; i<kernelVolume; ++i){
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    if  (device == torch::kCPU) {
      continue;
    }
    else{
      functor::SummaryRFForwardFunctor<tv::GPU, int> forwardFtor;
      forwardFtor(
        tv::TorchGPU(),
        tv::torch2tv<const int>(indicePairs).subview(i),
        tv::torch2tv<int>(summarRFs),
        nHot
      );
      // printf("summarRFs[0]: %d \n", tv::torch2tv<int>(summarRFs)[0]);
    TV_CHECK_CUDA_ERR();
    }
  }
  return summarRFs;

}

template <typename T>
torch::Tensor indiceAvgPool(torch::Tensor features, torch::Tensor indicePairs,
                          torch::Tensor indiceNum, int64_t numAct, torch::Tensor summaryrf) {
  auto device = features.device().type();
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numAct, numInPlanes}, options);
  double totalTime = 0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    if (device == torch::kCPU) {
      // functor::SparseAvgPoolForwardFunctor<tv::CPU, T, int> forwardFtor;
      // forwardFtor(tv::CPU(), tv::torch2tv<T>(output),
      //             tv::torch2tv<const T>(features),
      //             tv::torch2tv<const int>(indicePairs).subview(i), nHot);
      return output;
    } else {
      functor::SparseAvgPoolForwardFunctor<tv::GPU, T, int> forwardFtor;
      forwardFtor(tv::TorchGPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot,
                  tv::torch2tv<const int>(summaryrf));
      TV_CHECK_CUDA_ERR();
    }
    // totalTime += timer.report() / 1000.0;
  }
  // std::cout << "maxpool forward time " << totalTime << std::endl;
  return output;
}

template <typename T>
torch::Tensor indiceAvgPoolBackward(torch::Tensor features,
                                  torch::Tensor outFeatures,
                                  torch::Tensor outGrad, torch::Tensor indicePairs,
                                  torch::Tensor indiceNum, torch::Tensor summaryrf) {
  auto device = features.device().type();
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
    auto kernelVolume = indicePairs.size(0);
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    if (device == torch::kCPU) {
      // functor::SparseAvgPoolBackwardFunctor<tv::CPU, T, int> backwardFtor;
      // backwardFtor(tv::CPU(), tv::torch2tv<const T>(outFeatures),
      //              tv::torch2tv<const T>(features),
      //              tv::torch2tv<const T>(outGrad), tv::torch2tv<T>(inputGrad),
      //              tv::torch2tv<const int>(indicePairs).subview(i), nHot,
      //              tv::torch2tv<const int>(summaryrf));
      return inputGrad;
    } else {
      functor::SparseAvgPoolBackwardFunctor<tv::GPU, T, int> backwardFtor;
      backwardFtor(tv::TorchGPU(), tv::torch2tv<const T>(outFeatures),
                   tv::torch2tv<const T>(features),
                   tv::torch2tv<const T>(outGrad), tv::torch2tv<T>(inputGrad),
                   tv::torch2tv<const int>(indicePairs).subview(i), nHot,
                   tv::torch2tv<const int>(summaryrf));
      TV_CHECK_CUDA_ERR();
    }
  }
  return inputGrad;
}
} // namespace spconv

#endif