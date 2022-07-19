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

#include <ATen/ATen.h>
#include <chrono>
#include <limits>
#include <spconv/summaryRF.h>
#include <spconv/mp_helper.h>
#include <tensorview/helper_kernel.cu.h>
#include <tensorview/helper_launch.h>
#include <tensorview/tensorview.h>
#include <type_traits>

namespace spconv {
template <typename T>
__global__ void summaryRFFwdKernel(const T *indicesIn,
                                   const T *indicesOut, 
                                   int size,
                                   T *num_RF) {
  // if (threadIdx.x == 1){
  //   printf("blockDim.x: %d, gridDim.x: %d \n", blockDim.x, gridDim.x);
  // }
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for(int i = idx; i<size; i+=blockDim.x * gridDim.x){
    // if (threadIdx.x == 1){
    //   printf("iter once \n");
    // }
    num_RF[indicesOut[i]] += 1;
  }
  }

namespace functor {
template <typename T>
struct SummaryRFForwardFunctor<tv::GPU, T> {
  void operator()(  const tv::GPU &d, 
                    tv::TensorView<const T> indices, 
                    tv::TensorView<T> num_RF, 
                    int size
                ) {
    int num_thread = 256;
    int num_iter   = 8;
    int num_block  = size / (256*num_iter)+1;
    // printf("size: %d, num_iter: %d, num_block: %d", size, num_iter, num_block);
    if (size<=0){
      return;
    }
    else{
      spconv::summaryRFFwdKernel<T>
                          <<<dim3(num_block, 1), dim3(num_thread, 1)>>>
                            (indices.subview(0).data(), 
                             indices.subview(1).data(),
                             size,
                             num_RF.data());
      TV_CHECK_CUDA_ERR();
    }
  }
};
} // namespace functor

#define DECLARE_GPU_SPECS_T_INDEX(T) \
  template struct functor::SummaryRFForwardFunctor<tv::GPU, T>;

#define DECLARE_GPU_SPECS() DECLARE_GPU_SPECS_T_INDEX(int);

// DECLARE_GPU_SPECS(float);
// DECLARE_GPU_SPECS(double);
// DECLARE_GPU_SPECS(at::Half);
DECLARE_GPU_SPECS();

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_T_INDEX
} // namespace spconv