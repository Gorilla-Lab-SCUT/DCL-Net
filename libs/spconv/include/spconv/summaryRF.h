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

#ifndef SPARSE_SUMMARYRF_FUNCTOR_H_
#define SPARSE_SUMMARYRF_FUNCTOR_H_
#include <tensorview/tensorview.h>

namespace spconv
{
namespace functor
{
template <typename Device, typename Index>
struct SummaryRFForwardFunctor
{
    void operator()(const Device& d, 
                    tv::TensorView<const Index> indices, 
                    tv::TensorView<Index> num_RF, 
                    int size);
};


} // namespace functor
} // namespace spconv

#endif