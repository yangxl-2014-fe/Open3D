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

#include <benchmark/benchmark.h>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/PointCloud.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <omp.h>
#include <cmath>
#include <functional>
#include <vector>

namespace open3d {
namespace t {
namespace geometry {


static void ComputePosePointToPlaneTBB(const float *source_points_ptr,
                                const float *target_points_ptr,
                                const float *target_normals_ptr,
                                const int64_t *correspondence_first,
                                const int64_t *correspondence_second,
                                const int n,
                                core::Tensor &pose,
                                const core::Dtype &dtype,
                                const core::Device &device) {
 
    // ATA is a {6,6} symmetric matrix, so we only need 21 elements instead of 36.
    // ATB is of shape {6,1}. Combining both, A_1x27 is a temp. storage 
    // with [0:21] elements as of ATA and [21:27] elements as of ATB.

    // Identity element for running_total reduction variable: zeros_27.
    std::vector<double> zeros_27(27, 0.0);
    std::vector<double> A_1x27_vec = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_27,
            [&](tbb::blocked_range<int> r, std::vector<double> running_total) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
                    const int64_t &source_index =
                            3 * correspondence_first[workload_idx];
                    const int64_t &target_index =
                            3 * correspondence_second[workload_idx];

                    const float &sx = (source_points_ptr[source_index + 0]);
                    const float &sy = (source_points_ptr[source_index + 1]);
                    const float &sz = (source_points_ptr[source_index + 2]);
                    const float &tx = (target_points_ptr[target_index + 0]);
                    const float &ty = (target_points_ptr[target_index + 1]);
                    const float &tz = (target_points_ptr[target_index + 2]);
                    const float &nx = (target_normals_ptr[target_index + 0]);
                    const float &ny = (target_normals_ptr[target_index + 1]);
                    const float &nz = (target_normals_ptr[target_index + 2]);

                    const double bi =
                            (tx - sx) * nx + (ty - sy) * ny + (tz - sz) * nz;
                    const double ai[] = {(nz * sy - ny * sz),
                                        (nx * sz - nz * sx),
                                        (ny * sx - nx * sy),
                                        nx,
                                        ny,
                                        nz};

                    for (int i = 0, j = 0; j < 6; j++) {
                        for (int k = 0; k <= j; k++) {
                            // ATA {N,21}
                            running_total[i] += ai[j] * ai[k];
                            i++;
                        }
                        // ATB {N,6}.
                        running_total[21 + j] += ai[j] * bi;
                    }
                }
                return running_total;
            },
            [&](std::vector<double> a, std::vector<double> b) {
                std::vector<double> result(27);
                for (int j = 0; j < 27; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });

    core::Tensor ATA =
            core::Tensor::Empty({6, 6}, core::Dtype::Float64, device);
    double *ata_ptr = ATA.GetDataPtr<double>();

    core::Tensor ATB =
            core::Tensor::Empty({6, 1}, core::Dtype::Float64, device);
    double *atb_ptr = ATB.GetDataPtr<double>();

    // ATA {1,21} to ATA {6,6}.
    for (int i = 0, j = 0; j < 6; j++) {
        for (int k = 0; k <= j; k++) {
            ata_ptr[j * 6 + k] = A_1x27_vec[i];
            ata_ptr[k * 6 + j] = A_1x27_vec[i];
            i++;
        }
        atb_ptr[j] = A_1x27_vec[j + 21];
    }

    // ATA(6,6) . Pose(6,1) = ATB(6,1)
    pose = ATA.Solve(ATB).Reshape({-1}).To(dtype);
}

/*
static void ComputePosePointToPlaneOpenMP(const float *source_points_ptr,
                                    const float *target_points_ptr,
                                    const float *target_normals_ptr,
                                    const int64_t *correspondence_first,
                                    const int64_t *correspondence_second,
                                    const int n,
                                    core::Tensor &pose,
                                    const core::Dtype &dtype,
                                    const core::Device &device) {
    // ATA is a {6,6} symmetric matrix, so we only need 21 elements instead of 36.
    // ATB is of shape {6,1}. Combining both, A_1x27 is a temp. storage 
    // with [0:21] elements as of ATA and [21:27] elements as of ATB.
    core::Tensor A_1x27 =
            core::Tensor::Zeros({1, 21}, core::Dtype::Float64, device);
    double *A_1x27_ptr = A_1x27.GetDataPtr<double>();

#pragma omp parallel for reduction(+ : A_1x27_ptr[:27])
    for (int64_t workload_idx = 0; workload_idx < n; ++workload_idx) {
        const int64_t &source_index = 3 * correspondence_first[workload_idx];
        const int64_t &target_index = 3 * correspondence_second[workload_idx];

        const float &sx = (source_points_ptr[source_index + 0]);
        const float &sy = (source_points_ptr[source_index + 1]);
        const float &sz = (source_points_ptr[source_index + 2]);
        const float &tx = (target_points_ptr[target_index + 0]);
        const float &ty = (target_points_ptr[target_index + 1]);
        const float &tz = (target_points_ptr[target_index + 2]);
        const float &nx = (target_normals_ptr[target_index + 0]);
        const float &ny = (target_normals_ptr[target_index + 1]);
        const float &nz = (target_normals_ptr[target_index + 2]);

        const double bi =
                (tx - sx) * nx + (ty - sy) * ny + (tz - sz) * nz;
        const double ai[] = {(nz * sy - ny * sz),
                            (nx * sz - nz * sx),
                            (ny * sx - nx * sy),
                            nx,
                            ny,
                            nz};

        for (int i = 0, j = 0; j < 6; j++) {
            for (int k = 0; k <= j; k++) {
                // ATA_ {1,21}, as ATA {6,6} is a symmetric matrix.
                A_1x27_ptr[i] += ai[j] * ai[k];
                i++;
            }
            // ATB {6,1}.
            A_1x27_ptr[21 + j] +=  ai[j] * bi;
        }
    }
    core::Tensor ATA =
            core::Tensor::Empty({6, 6}, core::Dtype::Float64, device);
    double *ata_ptr = ATA.GetDataPtr<double>();

    core::Tensor ATB =
            core::Tensor::Empty({6, 1}, core::Dtype::Float64, device);
    double *atb_ptr = ATB.GetDataPtr<double>();

    // ATA_ {1,21} to ATA {6,6}.
    for (int i = 0, j = 0; j < 6; j++) {
        for (int k = 0; k <= j; k++) {
            ata_ptr[j * 6 + k] = A_1x27_ptr[i];
            ata_ptr[k * 6 + j] = A_1x27_ptr[i];
            i++;
        }
        atb_ptr[j] = A_1x27_ptr[21 + j];
    }

    // ATA(6,6) . Pose(6,1) = ATB(6,1)
    pose = ATA.Solve(ATB).Reshape({-1}).To(dtype);
}

void BenchmarkComputePosePointToPlaneOpenMP(benchmark::State &state, const core::Device &device) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor source_points_contiguous = core::Tensor::Load(
            TEST_DATA_DIR "/icp_benchmark/source_points.npy");
    core::Tensor target_points_contiguous = core::Tensor::Load(
            TEST_DATA_DIR "/icp_benchmark/target_points.npy");
    core::Tensor target_normals_contiguous = core::Tensor::Load(
            TEST_DATA_DIR "/icp_benchmark/target_normals.npy");
    core::Tensor corres_first_contiguous =
            core::Tensor::Load(TEST_DATA_DIR "/icp_benchmark/corres_first.npy");
    core::Tensor corres_second_contiguous = core::Tensor::Load(
            TEST_DATA_DIR "/icp_benchmark/corres_second.npy");
    core::Tensor pose = core::Tensor::Empty({6}, dtype, device);
    int n = 144132;

    // Warm up.
    ComputePosePointToPlaneOpenMP(source_points_contiguous.GetDataPtr<float>(),
                               target_points_contiguous.GetDataPtr<float>(),
                               target_normals_contiguous.GetDataPtr<float>(),
                               corres_first_contiguous.GetDataPtr<int64_t>(),
                               corres_second_contiguous.GetDataPtr<int64_t>(),
                               n, pose, dtype, device);

    for (auto _ : state) {
        ComputePosePointToPlaneOpenMP(
                source_points_contiguous.GetDataPtr<float>(),
                target_points_contiguous.GetDataPtr<float>(),
                target_normals_contiguous.GetDataPtr<float>(),
                corres_first_contiguous.GetDataPtr<int64_t>(),
                corres_second_contiguous.GetDataPtr<int64_t>(), n, pose, dtype,
                device);
    }
}
*/
void BenchmarkComputePosePointToPlaneTBB(benchmark::State &state, const core::Device &device) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor source_points_contiguous = core::Tensor::Load(
            TEST_DATA_DIR "/icp_benchmark/source_points.npy");
    core::Tensor target_points_contiguous = core::Tensor::Load(
            TEST_DATA_DIR "/icp_benchmark/target_points.npy");
    core::Tensor target_normals_contiguous = core::Tensor::Load(
            TEST_DATA_DIR "/icp_benchmark/target_normals.npy");
    core::Tensor corres_first_contiguous =
            core::Tensor::Load(TEST_DATA_DIR "/icp_benchmark/corres_first.npy");
    core::Tensor corres_second_contiguous = core::Tensor::Load(
            TEST_DATA_DIR "/icp_benchmark/corres_second.npy");
    core::Tensor pose = core::Tensor::Empty({6}, dtype, device);
    int n = 144132;

    // Warm up.
    ComputePosePointToPlaneTBB(source_points_contiguous.GetDataPtr<float>(),
                               target_points_contiguous.GetDataPtr<float>(),
                               target_normals_contiguous.GetDataPtr<float>(),
                               corres_first_contiguous.GetDataPtr<int64_t>(),
                               corres_second_contiguous.GetDataPtr<int64_t>(),
                               n, pose, dtype, device);

    for (auto _ : state) {
        ComputePosePointToPlaneTBB(
                source_points_contiguous.GetDataPtr<float>(),
                target_points_contiguous.GetDataPtr<float>(),
                target_normals_contiguous.GetDataPtr<float>(),
                corres_first_contiguous.GetDataPtr<int64_t>(),
                corres_second_contiguous.GetDataPtr<int64_t>(), n, pose, dtype,
                device);
    }
}

// BENCHMARK_CAPTURE(BenchmarkComputePosePointToPlaneOpenMP, CPU, core::Device("CPU:0"))
//         ->Unit(benchmark::kMillisecond);


BENCHMARK_CAPTURE(BenchmarkComputePosePointToPlaneTBB, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
