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

namespace open3d {
namespace t {
namespace geometry {

void ICPBenchmark(benchmark::State& state, const core::Device& device) {
    int64_t num_points = 1000000;  // 1M
    PointCloud pcd(device);
    pcd.SetPoints(core::Tensor({num_points, 3}, core::Dtype::Float32, device));
    pcd.SetPointColors(
            core::Tensor({num_points, 3}, core::Dtype::Float32, device));

    // Warm up.
    open3d::geometry::PointCloud legacy_pcd = pcd.ToLegacyPointCloud();
    (void)legacy_pcd;

    for (auto _ : state) {
        open3d::geometry::PointCloud legacy_pcd = pcd.ToLegacyPointCloud();
    }
}

BENCHMARK_CAPTURE(ICPBenchmark, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
