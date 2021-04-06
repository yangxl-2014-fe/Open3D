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

#include "open3d/t/pipelines/registration/Registration.h"

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

static RegistrationResult GetRegistrationResultAndCorrespondences(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        open3d::core::nns::NearestNeighborSearch &target_nns,
        double max_correspondence_distance,
        const core::Tensor &transformation) {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }
    transformation.AssertShape({4, 4});
    transformation.AssertDtype(dtype);

    core::Tensor transformation_device = transformation.To(device);

    RegistrationResult result(transformation_device);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    bool check = target_nns.HybridIndex(max_correspondence_distance);
    if (!check) {
        utility::LogError(
                "[Tensor: EvaluateRegistration: "
                "GetRegistrationResultAndCorrespondences: "
                "NearestNeighborSearch::HybridSearch] "
                "Index is not set.");
    }

    core::Tensor distances;
    std::tie(result.correspondence_set_.first,
             result.correspondence_set_.second, distances) =
            target_nns.Hybrid1NNSearch(source.GetPoints(),
                                       max_correspondence_distance);

    // Number of good correspondences (C).
    int num_correspondences = result.correspondence_set_.first.GetLength();

    // Reduction sum of "distances" for error.
    double squared_error =
            static_cast<double>(distances.Sum({0}).Item<float>());
    result.fitness_ = static_cast<double>(num_correspondences) /
                      static_cast<double>(source.GetPoints().GetLength());
    result.inlier_rmse_ =
            std::sqrt(squared_error / static_cast<double>(num_correspondences));
    result.transformation_ = transformation;

    return result;
}

RegistrationResult EvaluateRegistration(const geometry::PointCloud &source,
                                        const geometry::PointCloud &target,
                                        double max_correspondence_distance,
                                        const core::Tensor &transformation) {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }
    transformation.AssertShape({4, 4});
    transformation.AssertDtype(dtype);
    core::Tensor transformation_device = transformation.To(device);

    open3d::core::nns::NearestNeighborSearch target_nns(target.GetPoints());

    geometry::PointCloud source_transformed = source.Clone();
    source_transformed.Transform(transformation_device);
    return GetRegistrationResultAndCorrespondences(
            source_transformed, target, target_nns, max_correspondence_distance,
            transformation_device);
}

RegistrationResult RegistrationICP(const geometry::PointCloud &source,
                                   const geometry::PointCloud &target,
                                   double max_correspondence_distance,
                                   const core::Tensor &init,
                                   const TransformationEstimation &estimation,
                                   const ICPConvergenceCriteria &criteria) {
    return RegistrationICPMultiScale(source, target, {-1}, {criteria},
                                     {max_correspondence_distance}, init,
                                     estimation);
}

RegistrationResult RegistrationICPMultiScale(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<double> &voxel_sizes,
        const std::vector<ICPConvergenceCriteria> &criterias,
        const std::vector<double> &max_correspondence_distances,
        const core::Tensor &init,
        const TransformationEstimation &estimation) {
    core::Device device = source.GetDevice();
    core::Dtype dtype = source.GetPoints().GetDtype();
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    int64_t num_iterations = int64_t(criterias.size());
    if (!(criterias.size() == voxel_sizes.size() &&
          criterias.size() == max_correspondence_distances.size())) {
        utility::LogError(
                " [RegistrationICPMultiScale]: Size of ICPConvergenceCriteria,"
                " voxel_size, max_correspondence_distances vectors must be "
                "same.");
    }

    for (int64_t i = 1; i < num_iterations; i++) {
        if (voxel_sizes[i] >= voxel_sizes[i - 1]) {
            utility::LogError(
                    " [MultiScaleICP] Voxel sizes must be in strictly "
                    "decreasing "
                    "order.");
        }
        if (max_correspondence_distances[i] <= 0.0) {
            utility::LogError(
                    " Max correspondence distance must be greater than 0, but"
                    " got {} in scale: {}.",
                    max_correspondence_distances[i], i);
        }
    }

    init.AssertShape({4, 4});
    init.AssertDtype(dtype);

    core::Tensor transformation_device = init.To(device);

    std::vector<t::geometry::PointCloud> source_down_pyramid(num_iterations);
    std::vector<t::geometry::PointCloud> target_down_pyramid(num_iterations);

    if (voxel_sizes[num_iterations - 1] == -1) {
        source_down_pyramid[num_iterations - 1] = source.Clone();
        target_down_pyramid[num_iterations - 1] = target;
    } else {
        source_down_pyramid[num_iterations - 1] =
                source.Clone().VoxelDownSample(voxel_sizes[num_iterations - 1]);
        target_down_pyramid[num_iterations - 1] =
                target.Clone().VoxelDownSample(voxel_sizes[num_iterations - 1]);
    }

    for (int k = num_iterations - 2; k >= 0; k--) {
        source_down_pyramid[k] =
                source_down_pyramid[k + 1].VoxelDownSample(voxel_sizes[k]);
        target_down_pyramid[k] =
                target_down_pyramid[k + 1].VoxelDownSample(voxel_sizes[k]);
    }

    RegistrationResult result(transformation_device);

    double prev_fitness_ = 0;
    double prev_inliner_rmse_ = 0;

    for (int64_t i = 0; i < num_iterations; i++) {
        source_down_pyramid[i].Transform(transformation_device);

        core::nns::NearestNeighborSearch target_nns(
                target_down_pyramid[i].GetPoints());

        result = GetRegistrationResultAndCorrespondences(
                source_down_pyramid[i], target_down_pyramid[i], target_nns,
                max_correspondence_distances[i], transformation_device);

        for (int j = 0; j < criterias[i].max_iteration_; j++) {
            utility::LogDebug(
                    "Scale #{:d} ICP Iteration #{:d}: Fitness {:.4f}, RMSE "
                    "{:.4f}",
                    i + 1, j, result.fitness_, result.inlier_rmse_);

            core::Tensor update = estimation.ComputeTransformation(
                    source_down_pyramid[i], target_down_pyramid[i],
                    result.correspondence_set_);

            // Multiply the transform to the cumulative transformation (update).
            transformation_device = update.Matmul(transformation_device);
            // Apply the transform on source pointcloud.
            source_down_pyramid[i].Transform(update);

            prev_fitness_ = result.fitness_;
            prev_inliner_rmse_ = result.inlier_rmse_;

            result = GetRegistrationResultAndCorrespondences(
                    source_down_pyramid[i], target_down_pyramid[i], target_nns,
                    max_correspondence_distances[i], transformation_device);

            // ICPConvergenceCriteria, to terminate iteration.
            if (j != 0 &&
                std::abs(prev_fitness_ - result.fitness_) <
                        criterias[i].relative_fitness_ &&
                std::abs(prev_inliner_rmse_ - result.inlier_rmse_) <
                        criterias[i].relative_rmse_) {
                break;
            }
        }
    }
    return result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
