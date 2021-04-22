// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <cuda.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/CoreUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/pipelines/kernel/ComputeTransformImpl.h"
#include "open3d/t/pipelines/kernel/Reduction6x6Impl.cuh"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

__global__ void ComputePosePointToPlaneCUDAKernel(
        const float *source_points_ptr,
        const float *target_points_ptr,
        const float *target_normals_ptr,
        const int64_t *correspondences_first,
        const int64_t *correspondences_second,
        const int n,
        float *global_sum) {
    const int THREAD_1D_UNIT = 256;
    __shared__ float local_sum0[THREAD_1D_UNIT];
    __shared__ float local_sum1[THREAD_1D_UNIT];
    __shared__ float local_sum2[THREAD_1D_UNIT];
    __shared__ float local_sum3[THREAD_1D_UNIT];
    __shared__ float local_sum4[THREAD_1D_UNIT];
    __shared__ float local_sum5[THREAD_1D_UNIT];
    __shared__ float local_sum6[THREAD_1D_UNIT];
    __shared__ float local_sum7[THREAD_1D_UNIT];
    __shared__ float local_sum8[THREAD_1D_UNIT];
    __shared__ float local_sum9[THREAD_1D_UNIT];
    __shared__ float local_sum10[THREAD_1D_UNIT];
    __shared__ float local_sum11[THREAD_1D_UNIT];
    __shared__ float local_sum12[THREAD_1D_UNIT];
    __shared__ float local_sum13[THREAD_1D_UNIT];
    __shared__ float local_sum14[THREAD_1D_UNIT];
    __shared__ float local_sum15[THREAD_1D_UNIT];
    __shared__ float local_sum16[THREAD_1D_UNIT];
    __shared__ float local_sum17[THREAD_1D_UNIT];
    __shared__ float local_sum18[THREAD_1D_UNIT];
    __shared__ float local_sum19[THREAD_1D_UNIT];
    __shared__ float local_sum20[THREAD_1D_UNIT];
    __shared__ float local_sum21[THREAD_1D_UNIT];
    __shared__ float local_sum22[THREAD_1D_UNIT];
    __shared__ float local_sum23[THREAD_1D_UNIT];
    __shared__ float local_sum24[THREAD_1D_UNIT];
    __shared__ float local_sum25[THREAD_1D_UNIT];
    __shared__ float local_sum26[THREAD_1D_UNIT];
    __shared__ float local_sum27[THREAD_1D_UNIT];
    __shared__ float local_sum28[THREAD_1D_UNIT];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    if (tid >= n) return;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;
    local_sum3[tid] = 0;
    local_sum4[tid] = 0;
    local_sum5[tid] = 0;
    local_sum6[tid] = 0;
    local_sum7[tid] = 0;
    local_sum8[tid] = 0;
    local_sum9[tid] = 0;
    local_sum10[tid] = 0;
    local_sum11[tid] = 0;
    local_sum12[tid] = 0;
    local_sum13[tid] = 0;
    local_sum14[tid] = 0;
    local_sum15[tid] = 0;
    local_sum16[tid] = 0;
    local_sum17[tid] = 0;
    local_sum18[tid] = 0;
    local_sum19[tid] = 0;
    local_sum20[tid] = 0;
    local_sum21[tid] = 0;
    local_sum22[tid] = 0;
    local_sum23[tid] = 0;
    local_sum24[tid] = 0;
    local_sum25[tid] = 0;
    local_sum26[tid] = 0;
    local_sum27[tid] = 0;
    local_sum28[tid] = 0;

    float J[6] = {0}, reduction[21 + 6];
    float r = 0;

    bool valid = GetJacobianPointToPlane(
            tid, source_points_ptr, target_points_ptr, target_normals_ptr,
            correspondences_first, correspondences_second, J, r);

    // Dump J, r into JtJ and Jtr
    int offset = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            reduction[offset++] = J[i] * J[j];
        }
    }
    for (int i = 0; i < 6; ++i) {
        reduction[offset++] = J[i] * r;
    }
    // reduction[offset++] = r * r;
    // reduction[offset++] = valid;

    ReduceSum6x6LinearSystem<float, THREAD_1D_UNIT>(
            tid, valid, reduction, local_sum0, local_sum1, local_sum2,
            local_sum3, local_sum4, local_sum5, local_sum6, local_sum7,
            local_sum8, local_sum9, local_sum10, local_sum11, local_sum12,
            local_sum13, local_sum14, local_sum15, local_sum16, local_sum17,
            local_sum18, local_sum19, local_sum20, local_sum21, local_sum22,
            local_sum23, local_sum24, local_sum25, local_sum26, local_sum27,
            local_sum28, global_sum,
            /*reduce_residual = */ false);
}

void ComputePosePointToPlaneCUDA(const float *source_points_ptr,
                                 const float *target_points_ptr,
                                 const float *target_normals_ptr,
                                 const int64_t *correspondences_first,
                                 const int64_t *correspondences_second,
                                 const int n,
                                 core::Tensor &pose,
                                 const core::Dtype &dtype,
                                 const core::Device &device) {
    core::Tensor global_sum =
            core::Tensor::Zeros({27}, core::Dtype::Float32, device);
    float *global_sum_ptr = global_sum.GetDataPtr<float>();

    const int THREAD_1D_UNIT = 256;
    const dim3 blocks((n + THREAD_1D_UNIT - 1) / THREAD_1D_UNIT);
    const dim3 threads(THREAD_1D_UNIT);

    ComputePosePointToPlaneCUDAKernel<<<blocks, threads>>>(
            source_points_ptr, target_points_ptr, target_normals_ptr,
            correspondences_first, correspondences_second, n, global_sum_ptr);

    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    DecodeAndSolve6x6(global_sum, pose);
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
