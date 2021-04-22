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

// Private header. Do not include in Open3d.h.

#pragma once

#include <cmath>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/CoreUtil.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

template <typename T>
__device__ inline void WarpReduceSum(volatile T* local_sum, const int tid) {
    local_sum[tid] += local_sum[tid + 32];
    local_sum[tid] += local_sum[tid + 16];
    local_sum[tid] += local_sum[tid + 8];
    local_sum[tid] += local_sum[tid + 4];
    local_sum[tid] += local_sum[tid + 2];
    local_sum[tid] += local_sum[tid + 1];
}

template <typename T, size_t BLOCK_SIZE>
__device__ inline void BlockReduceSum(const int tid, volatile T* local_sum) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum[tid] += local_sum[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum[tid] += local_sum[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum[tid] += local_sum[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        WarpReduceSum<T>(local_sum, tid);
    }
}

template <typename T, size_t BLOCK_SIZE>
__device__ inline void BlockReduceSum(const int tid,
                                      volatile T* local_sum0,
                                      volatile T* local_sum1) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum0[tid] += local_sum0[tid + 256];
            local_sum1[tid] += local_sum1[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceSum<float>(local_sum0, tid);
        WarpReduceSum<float>(local_sum1, tid);
    }
}

template <typename T, size_t BLOCK_SIZE>
__device__ inline void BlockReduceSum(const int tid,
                                      volatile T* local_sum0,
                                      volatile T* local_sum1,
                                      volatile T* local_sum2) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum0[tid] += local_sum0[tid + 256];
            local_sum1[tid] += local_sum1[tid + 256];
            local_sum2[tid] += local_sum2[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
            local_sum2[tid] += local_sum2[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
            local_sum2[tid] += local_sum2[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceSum<float>(local_sum0, tid);
        WarpReduceSum<float>(local_sum1, tid);
        WarpReduceSum<float>(local_sum2, tid);
    }
}

template <typename T, size_t BLOCK_SIZE>
__device__ inline void BlockReduceSum(const int tid,
                                      volatile T* local_sum0,
                                      volatile T* local_sum1,
                                      volatile T* local_sum2,
                                      volatile T* local_sum3,
                                      volatile T* local_sum4,
                                      volatile T* local_sum5,
                                      volatile T* local_sum6,
                                      volatile T* local_sum7,
                                      volatile T* local_sum8,
                                      volatile T* local_sum9,
                                      volatile T* local_sum10,
                                      volatile T* local_sum11,
                                      volatile T* local_sum12,
                                      volatile T* local_sum13,
                                      volatile T* local_sum14,
                                      volatile T* local_sum15,
                                      volatile T* local_sum16,
                                      volatile T* local_sum17,
                                      volatile T* local_sum18,
                                      volatile T* local_sum19,
                                      volatile T* local_sum20,
                                      volatile T* local_sum21,
                                      volatile T* local_sum22,
                                      volatile T* local_sum23,
                                      volatile T* local_sum24,
                                      volatile T* local_sum25,
                                      volatile T* local_sum26) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum0[tid] += local_sum0[tid + 256];
            local_sum1[tid] += local_sum1[tid + 256];
            local_sum2[tid] += local_sum2[tid + 256];
            local_sum3[tid] += local_sum3[tid + 256];
            local_sum4[tid] += local_sum4[tid + 256];
            local_sum5[tid] += local_sum5[tid + 256];
            local_sum6[tid] += local_sum6[tid + 256];
            local_sum7[tid] += local_sum7[tid + 256];
            local_sum8[tid] += local_sum8[tid + 256];
            local_sum9[tid] += local_sum9[tid + 256];
            local_sum10[tid] += local_sum10[tid + 256];
            local_sum11[tid] += local_sum11[tid + 256];
            local_sum12[tid] += local_sum12[tid + 256];
            local_sum13[tid] += local_sum13[tid + 256];
            local_sum14[tid] += local_sum14[tid + 256];
            local_sum15[tid] += local_sum15[tid + 256];
            local_sum16[tid] += local_sum16[tid + 256];
            local_sum17[tid] += local_sum17[tid + 256];
            local_sum18[tid] += local_sum18[tid + 256];
            local_sum19[tid] += local_sum19[tid + 256];
            local_sum20[tid] += local_sum20[tid + 256];
            local_sum21[tid] += local_sum21[tid + 256];
            local_sum22[tid] += local_sum22[tid + 256];
            local_sum23[tid] += local_sum23[tid + 256];
            local_sum24[tid] += local_sum24[tid + 256];
            local_sum25[tid] += local_sum25[tid + 256];
            local_sum26[tid] += local_sum26[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
            local_sum2[tid] += local_sum2[tid + 128];
            local_sum3[tid] += local_sum3[tid + 128];
            local_sum4[tid] += local_sum4[tid + 128];
            local_sum5[tid] += local_sum5[tid + 128];
            local_sum6[tid] += local_sum6[tid + 128];
            local_sum7[tid] += local_sum7[tid + 128];
            local_sum8[tid] += local_sum8[tid + 128];
            local_sum9[tid] += local_sum9[tid + 128];
            local_sum10[tid] += local_sum10[tid + 128];
            local_sum11[tid] += local_sum11[tid + 128];
            local_sum12[tid] += local_sum12[tid + 128];
            local_sum13[tid] += local_sum13[tid + 128];
            local_sum14[tid] += local_sum14[tid + 128];
            local_sum15[tid] += local_sum15[tid + 128];
            local_sum16[tid] += local_sum16[tid + 128];
            local_sum17[tid] += local_sum17[tid + 128];
            local_sum18[tid] += local_sum18[tid + 128];
            local_sum19[tid] += local_sum19[tid + 128];
            local_sum20[tid] += local_sum20[tid + 128];
            local_sum21[tid] += local_sum21[tid + 128];
            local_sum22[tid] += local_sum22[tid + 128];
            local_sum23[tid] += local_sum23[tid + 128];
            local_sum24[tid] += local_sum24[tid + 128];
            local_sum25[tid] += local_sum25[tid + 128];
            local_sum26[tid] += local_sum26[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
            local_sum2[tid] += local_sum2[tid + 64];
            local_sum3[tid] += local_sum3[tid + 64];
            local_sum4[tid] += local_sum4[tid + 64];
            local_sum5[tid] += local_sum5[tid + 64];
            local_sum6[tid] += local_sum6[tid + 64];
            local_sum7[tid] += local_sum7[tid + 64];
            local_sum8[tid] += local_sum8[tid + 64];
            local_sum9[tid] += local_sum9[tid + 64];
            local_sum10[tid] += local_sum10[tid + 64];
            local_sum11[tid] += local_sum11[tid + 64];
            local_sum12[tid] += local_sum12[tid + 64];
            local_sum13[tid] += local_sum13[tid + 64];
            local_sum14[tid] += local_sum14[tid + 64];
            local_sum15[tid] += local_sum15[tid + 64];
            local_sum16[tid] += local_sum16[tid + 64];
            local_sum17[tid] += local_sum17[tid + 64];
            local_sum18[tid] += local_sum18[tid + 64];
            local_sum19[tid] += local_sum19[tid + 64];
            local_sum20[tid] += local_sum20[tid + 64];
            local_sum21[tid] += local_sum21[tid + 64];
            local_sum22[tid] += local_sum22[tid + 64];
            local_sum23[tid] += local_sum23[tid + 64];
            local_sum24[tid] += local_sum24[tid + 64];
            local_sum25[tid] += local_sum25[tid + 64];
            local_sum26[tid] += local_sum26[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceSum<float>(local_sum0, tid);
        WarpReduceSum<float>(local_sum1, tid);
        WarpReduceSum<float>(local_sum2, tid);
        WarpReduceSum<float>(local_sum3, tid);
        WarpReduceSum<float>(local_sum4, tid);
        WarpReduceSum<float>(local_sum5, tid);
        WarpReduceSum<float>(local_sum6, tid);
        WarpReduceSum<float>(local_sum7, tid);
        WarpReduceSum<float>(local_sum8, tid);
        WarpReduceSum<float>(local_sum9, tid);
        WarpReduceSum<float>(local_sum10, tid);
        WarpReduceSum<float>(local_sum11, tid);
        WarpReduceSum<float>(local_sum12, tid);
        WarpReduceSum<float>(local_sum13, tid);
        WarpReduceSum<float>(local_sum14, tid);
        WarpReduceSum<float>(local_sum15, tid);
        WarpReduceSum<float>(local_sum16, tid);
        WarpReduceSum<float>(local_sum17, tid);
        WarpReduceSum<float>(local_sum18, tid);
        WarpReduceSum<float>(local_sum19, tid);
        WarpReduceSum<float>(local_sum20, tid);
        WarpReduceSum<float>(local_sum21, tid);
        WarpReduceSum<float>(local_sum22, tid);
        WarpReduceSum<float>(local_sum23, tid);
        WarpReduceSum<float>(local_sum24, tid);
        WarpReduceSum<float>(local_sum25, tid);
        WarpReduceSum<float>(local_sum26, tid);
    }
}

template <typename T, size_t BLOCK_SIZE>
__device__ inline void ReduceSum6x6LinearSystem(const int tid,
                                                bool valid,
                                                const T* reduction,
                                                volatile T* local_sum0,
                                                volatile T* local_sum1,
                                                volatile T* local_sum2,
                                                volatile T* local_sum3,
                                                volatile T* local_sum4,
                                                volatile T* local_sum5,
                                                volatile T* local_sum6,
                                                volatile T* local_sum7,
                                                volatile T* local_sum8,
                                                volatile T* local_sum9,
                                                volatile T* local_sum10,
                                                volatile T* local_sum11,
                                                volatile T* local_sum12,
                                                volatile T* local_sum13,
                                                volatile T* local_sum14,
                                                volatile T* local_sum15,
                                                volatile T* local_sum16,
                                                volatile T* local_sum17,
                                                volatile T* local_sum18,
                                                volatile T* local_sum19,
                                                volatile T* local_sum20,
                                                volatile T* local_sum21,
                                                volatile T* local_sum22,
                                                volatile T* local_sum23,
                                                volatile T* local_sum24,
                                                volatile T* local_sum25,
                                                volatile T* local_sum26,
                                                volatile T* local_sum27,
                                                volatile T* local_sum28,
                                                T* global_sum,
                                                bool reduce_residual = true) {
    // Sum reduction: JtJ(21) and Jtr(6)
    if (valid) {
        local_sum0[tid] = reduction[0];
        local_sum1[tid] = reduction[1];
        local_sum2[tid] = reduction[2];
        local_sum3[tid] = reduction[3];
        local_sum4[tid] = reduction[4];
        local_sum5[tid] = reduction[5];
        local_sum6[tid] = reduction[6];
        local_sum7[tid] = reduction[7];
        local_sum8[tid] = reduction[8];
        local_sum9[tid] = reduction[9];
        local_sum10[tid] = reduction[10];
        local_sum11[tid] = reduction[11];
        local_sum12[tid] = reduction[12];
        local_sum13[tid] = reduction[13];
        local_sum14[tid] = reduction[14];
        local_sum15[tid] = reduction[15];
        local_sum16[tid] = reduction[16];
        local_sum17[tid] = reduction[17];
        local_sum18[tid] = reduction[18];
        local_sum19[tid] = reduction[19];
        local_sum20[tid] = reduction[20];
        local_sum20[tid] = reduction[20];
        local_sum21[tid] = reduction[21];
        local_sum22[tid] = reduction[22];
        local_sum23[tid] = reduction[23];
        local_sum24[tid] = reduction[24];
        local_sum25[tid] = reduction[25];
        local_sum26[tid] = reduction[26];
    } else {
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
        local_sum20[tid] = 0;
        local_sum21[tid] = 0;
        local_sum22[tid] = 0;
        local_sum23[tid] = 0;
        local_sum24[tid] = 0;
        local_sum25[tid] = 0;
        local_sum26[tid] = 0;
    }

    __syncthreads();

    BlockReduceSum<float, BLOCK_SIZE>(
            tid, local_sum0, local_sum1, local_sum2, local_sum3, local_sum4,
            local_sum5, local_sum6, local_sum7, local_sum8, local_sum9,
            local_sum10, local_sum11, local_sum12, local_sum13, local_sum14,
            local_sum15, local_sum16, local_sum17, local_sum18, local_sum19,
            local_sum20, local_sum21, local_sum22, local_sum23, local_sum24,
            local_sum25, local_sum26);

    if (tid == 0) {
        atomicAdd(&global_sum[0], local_sum0[0]);
        atomicAdd(&global_sum[1], local_sum1[0]);
        atomicAdd(&global_sum[2], local_sum2[0]);
        atomicAdd(&global_sum[3], local_sum3[0]);
        atomicAdd(&global_sum[4], local_sum4[0]);
        atomicAdd(&global_sum[5], local_sum5[0]);
        atomicAdd(&global_sum[6], local_sum6[0]);
        atomicAdd(&global_sum[7], local_sum7[0]);
        atomicAdd(&global_sum[8], local_sum8[0]);
        atomicAdd(&global_sum[9], local_sum9[0]);
        atomicAdd(&global_sum[10], local_sum10[0]);
        atomicAdd(&global_sum[11], local_sum11[0]);
        atomicAdd(&global_sum[12], local_sum12[0]);
        atomicAdd(&global_sum[13], local_sum13[0]);
        atomicAdd(&global_sum[14], local_sum14[0]);
        atomicAdd(&global_sum[15], local_sum15[0]);
        atomicAdd(&global_sum[16], local_sum16[0]);
        atomicAdd(&global_sum[17], local_sum17[0]);
        atomicAdd(&global_sum[18], local_sum18[0]);
        atomicAdd(&global_sum[19], local_sum19[0]);
        atomicAdd(&global_sum[20], local_sum20[0]);
        atomicAdd(&global_sum[21], local_sum21[0]);
        atomicAdd(&global_sum[22], local_sum22[0]);
        atomicAdd(&global_sum[23], local_sum23[0]);
        atomicAdd(&global_sum[24], local_sum24[0]);
        atomicAdd(&global_sum[25], local_sum25[0]);
        atomicAdd(&global_sum[26], local_sum26[0]);
    }
    __syncthreads();

    if (reduce_residual) {
        // Sum reduction: residual(1) and inlier(1)
        {
            local_sum0[tid] = valid ? reduction[27] : 0;
            local_sum1[tid] = valid ? reduction[28] : 0;
            __syncthreads();

            BlockReduceSum<float, BLOCK_SIZE>(tid, local_sum0, local_sum1);
            if (tid == 0) {
                atomicAdd(&global_sum[27], local_sum0[0]);
                atomicAdd(&global_sum[28], local_sum1[0]);
            }
            __syncthreads();
        }
    }
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
