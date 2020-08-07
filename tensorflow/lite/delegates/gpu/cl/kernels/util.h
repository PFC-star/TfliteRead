/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UTIL_H_

#include <string>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

std::string GetCommonDefines(CalculationsPrecision precision);

// Calculates correct X coordinate when stride != 1 and batch != 1 for layouts
// with B after W (for example HWBC4) and WB stored in one axis of GPU
// resources.
std::string GetXStrideCorrected(const std::string& src_x,
                                const std::string& batch_size,
                                const std::string& stride_x,
                                const std::string& padding_x);

template <DataType S, typename T>
void RearrangeWeightsToOHWIOGroupI4O4(
    const tflite::gpu::Tensor<OHWI, S>& weights, int out_group_size,
    absl::Span<T> dst) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);               // 32
  const int src_slices = DivideRoundUp(weights.shape.i, 4);               // 32
  const int kernel_x = weights.shape.w;                                   // 6
  const int kernel_y = weights.shape.h;                                   // 6

  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);       // 16 = 32 / 2
  // 即output设置了16个group, 每个group中包含了8个output filter
  // std::cout << dst_slices << " " << src_slices << " " << kernel_x << " " << kernel_y << " " << dst_groups << std::endl;

  // 将输入的128个filter分成了16组, 每组8个filter, 16个group中每个group又包含了两个小group, 每个小group有4个filter
  // 然后这4个filter被组成了一个c4
  // {ochannel group, kh, kw, ichannel_slice, small_group, ichannel_inner, ochannel_inner}
  int counter = 0;
  for (int d = 0; d < dst_groups; ++d) {                                  // 16
    for (int y = 0; y < kernel_y; ++y) {                                  // 6,  kh
      for (int x = 0; x < kernel_x; ++x) {                                // 6,  kw
        for (int s = 0; s < src_slices; ++s) {                            // 32, ic/4
          for (int d_group = 0; d_group < out_group_size; ++d_group) {    // 2
            for (int j = 0; j < 4; ++j) {                                 // 4
              T filter;                                                   // T是float4
              for (int i = 0; i < 4; ++i) {                               // 4
                const int s_ch = s * 4 + j;                               // 第s_ch个ichannel
                const int d_ch = (d * out_group_size + d_group) * 4 + i;  // 第d_ch个ochannel
                if (s_ch < weights.shape.i && d_ch < weights.shape.o) {   
                  const int f_index =
                      weights.shape.LinearIndex({d_ch, y, x, s_ch});
                  filter[i] = weights.data[f_index];
                } else {
                  filter[i] = 0.0f;
                }
              }
              // filter
              dst[counter++] = filter;            // 16 * 6 * 6 * 32 * 2 * 4 * 4
              // 可以这样理解，假如一次处理4个数，则就是读取了4个filter中的同一个点, 这样一次可以输出四个channel中的数，从而保证输出是C4的
              // 假如一次处理4*4个数，则读取的是4个filter中的4个channel的同一个点，考虑到输入也是C4的，则这4*4个数可以和输入中的一个C4做乘加计算
              // 这样输入输出访存仍然都是连续的
              // 这个outgroup即此处的2是一个硬件相关的参数，我的理解是此处的处理器可以同时处理两个4*4和1*4的向量矩阵乘
              // 把这4个filter的
            }
          }
        }
      }
    }
  }
}

// Returns fastest TextureAddressMode that return ZERO for out-of-range image
// coordinates.
//
// Unfortunately, CLK_ADDRESS_CLAMP is very slow on Adreno3xx and
// we can observe huge register overhead when compared to other modes.

// While using CLK_ADDRESS_NONE with out-of-range image coordinates is undefined
// in the OpenCL specification, we have observed that CLK_ADDRESS_NONE works
// like CLK_ADDRESS_CLAMP for out-of-range image coordinates for RGBA F16/F32
// textures on Adreno3xx devices. Using CLK_ADDRESS_NONE is significantly faster
// than CLK_ADDRESS_CLAMP on Adreno 3xx.
TextureAddressMode GetFastestZeroMode(const CLDevice& device);

// Returns float4 mask for last plane(batch of 4 channels)
// assumes that plane size is 4;
// for example we have 7 channels, in our data structures we align it to 8
// but 8s-channel will be empty, then last plane (batch of 4 channels) will
// have this mask (1, 1, 1, 0).
float4 GetMaskForLastPlane(int channels);

// returns first work group from wgs that has size not bigger than max_wg_size
// if no suitable groups among wgs, returns {1, 1, 1}
int3 GetFirstSuitableWorkGroup(const std::vector<int3>& wgs, int max_wg_size);

// task_size as amount of FLT4 processed elements.
int GetRecommendedBlockSizeForConv(const CLDevice& device,
                                   CalculationsPrecision precision,
                                   int task_size);
}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UTIL_H_
