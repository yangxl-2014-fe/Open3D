#include <fstream>
#include <sstream>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::t::pipelines::registration;

const int WIDTH = 1024;
const int HEIGHT = 768;

const Eigen::Vector3f CENTER_OFFSET(0.0f, 0.0f, -3.0f);
const std::string SRC_CLOUD = "source_pointcloud";
const std::string DST_CLOUD = "target_pointcloud";
const auto path_src = "../../../examples/test_data/ICP/cloud_bin_0.pcd";
const auto path_dst = "../../../examples/test_data/ICP/cloud_bin_2.pcd";

core::Device device = core::Device("CPU:0");
core::Dtype dtype = core::Dtype::Float32;

t::geometry::PointCloud PreProcessTensorPointCloud(t::geometry::PointCloud& pcd,
                                                   const core::Device& device,
                                                   const core::Dtype& dtype) {
    // Currently only Float32 pointcloud is supported.
    t::geometry::PointCloud pcd_copy(device);
    pcd_copy = pcd.To(device);

    for (std::string attr : {"points", "colors", "normals"}) {
        if (pcd.HasPointAttr(attr)) {
            pcd_copy.SetPointAttr(attr, pcd.GetPointAttr(attr).To(dtype));
        }
    }
    return pcd_copy;
}

class ICPExample {
public:
    ICPExample() {
        source_ = std::make_shared<geometry::PointCloud>();
        target_ = std::make_shared<geometry::PointCloud>();

        io::ReadPointCloud(path_src, *source_);
        if (source_->points_.empty()) {
            utility::LogError("Could not open {}", SRC_CLOUD);
            return;
        }
        io::ReadPointCloud(path_dst, *target_);
        if (target_->points_.empty()) {
            utility::LogError("Could not open {}", DST_CLOUD);
            return;
        }
    }

    void Run() {
        auto dtype_ = core::Dtype::Float32;
        auto device_ = core::Device("CPU:0");

        auto tsource_ = t::geometry::PointCloud::FromLegacyPointCloud(
                                *source_, dtype_, device_)
                                .VoxelDownSample(0.02);
        auto ttarget_ = t::geometry::PointCloud::FromLegacyPointCloud(
                                *target_, dtype_, device_)
                                .VoxelDownSample(0.02);

        auto transformation_device_ = core::Tensor::Eye(4, dtype_, device_);
        auto transformation_init_ = core::Tensor::Eye(4, dtype_, device_);

        auto DoSingleIterationICP = [&](visualization::visualizer::
                                                O3DVisualizer& o3dvis) {
            auto tsrc = tsource_.Clone();
            auto tdst = ttarget_.Clone();

            std::cout << " Transformation: "
                      << transformation_device_.ToString();

            std::vector<double> voxel_sizes = {0.05, 0.01};
            std::vector<ICPConvergenceCriteria> criterias = {
                    ICPConvergenceCriteria(0.0001, 0.0001, 30),
                    ICPConvergenceCriteria(0.0001, 0.0001, 30)};
            std::vector<double> max_correspondence_distances = {0.15, 0.3};
            core::Tensor init = transformation_init_;
            TransformationEstimationPointToPlane estimation;

            int64_t num_iterations = int64_t(criterias.size());
            auto transformation_device = init.To(device_);

            std::vector<t::geometry::PointCloud> source_down_pyramid(
                    num_iterations);
            std::vector<t::geometry::PointCloud> target_down_pyramid(
                    num_iterations);

            if (voxel_sizes[num_iterations - 1] == -1) {
                source_down_pyramid[num_iterations - 1] = tsrc;
                target_down_pyramid[num_iterations - 1] = tdst;
            } else {
                source_down_pyramid[num_iterations - 1] =
                        tsrc.Clone().VoxelDownSample(
                                voxel_sizes[num_iterations - 1]);
                target_down_pyramid[num_iterations - 1] =
                        tdst.Clone().VoxelDownSample(
                                voxel_sizes[num_iterations - 1]);
            }

            std::cout << " hey " << std::endl;

            for (int k = num_iterations - 2; k >= 0; k--) {
                source_down_pyramid[k] =
                        source_down_pyramid[k + 1].VoxelDownSample(
                                voxel_sizes[k]);
                target_down_pyramid[k] =
                        target_down_pyramid[k + 1].VoxelDownSample(
                                voxel_sizes[k]);
            }

            std::cout << " hey 2 " << std::endl;

            RegistrationResult result(transformation_device);

            for (int64_t i = 0; i < num_iterations; i++) {
                source_down_pyramid[i].Transform(transformation_device);

                std::cout << " hey 3 " << std::endl;

                core::nns::NearestNeighborSearch target_nns(
                        target_down_pyramid[i].GetPoints());

                result = GetRegistrationResultAndCorrespondences(
                        source_down_pyramid[i], target_down_pyramid[i],
                        target_nns, max_correspondence_distances[i],
                        transformation_device);

                std::cout << " hey 3 " << std::endl;

                for (int j = 0; j < criterias[i].max_iteration_; j++) {
                    utility::LogInfo(
                            " ICP Scale #{:d} Iteration #{:d}: Fitness {:.4f}, "
                            "RMSE "
                            "{:.4f}",
                            i + 1, j, result.fitness_, result.inlier_rmse_);

                    core::Tensor update = estimation.ComputeTransformation(
                            source_down_pyramid[i], target_down_pyramid[i],
                            result.correspondence_set_);

                    // Multiply the transform to the cumulative transformation
                    // (update).
                    transformation_device =
                            update.Matmul(transformation_device);
                    // Apply the transform on source pointcloud.
                    source_down_pyramid[i].Transform(update);

                    // auto source_legacy =
                    // std::make_shared<geometry::PointCloud>();
                    *source_ = source_down_pyramid[i].ToLegacyPointCloud();

                    o3dvis.RemoveGeometry(SRC_CLOUD);
                    o3dvis.AddGeometry(SRC_CLOUD, source_);

                    double prev_fitness_ = result.fitness_;
                    double prev_inliner_rmse_ = result.inlier_rmse_;

                    result = GetRegistrationResultAndCorrespondences(
                            source_down_pyramid[i], target_down_pyramid[i],
                            target_nns, max_correspondence_distances[i],
                            transformation_device);

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
        };

        visualization::Draw({visualization::DrawObject(SRC_CLOUD, source_),
                             visualization::DrawObject(DST_CLOUD, target_)},
                            "Open3D: ICP Registration example: Selection", 1024,
                            768, {{"ICP Registration ", DoSingleIterationICP}});
    }

private:
    RegistrationResult GetRegistrationResultAndCorrespondences(
            const t::geometry::PointCloud& source,
            const t::geometry::PointCloud& target,
            open3d::core::nns::NearestNeighborSearch& target_nns,
            double max_correspondence_distance,
            const core::Tensor& transformation) {
        core::Device device = source.GetDevice();
        core::Dtype dtype = core::Dtype::Float32;
        source.GetPoints().AssertDtype(dtype);
        target.GetPoints().AssertDtype(dtype);
        if (target.GetDevice() != device) {
            utility::LogError(
                    "Target Pointcloud device {} != Source Pointcloud's device "
                    "{}.",
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
        result.inlier_rmse_ = std::sqrt(
                squared_error / static_cast<double>(num_correspondences));
        result.transformation_ = transformation;

        return result;
    }

private:
    // core::Dtype dtype_;
    // core::Device device_;

    // core::Tensor transformation_device_;
    // core::Tensor transformation_init_;

    std::shared_ptr<geometry::PointCloud> source_;
    // Target PointCloud on CPU, used for visualization.
    std::shared_ptr<geometry::PointCloud> target_;

    // t::geometry::PointCloud tsource_;
    // t::geometry::PointCloud ttarget_;
};

int main(int argc, const char* argv[]) {
    // if (!utility::filesystem::DirectoryExists(path_source)) {
    //     utility::LogError(
    //             "This example needs to be run from the <build> "
    //             "directory");
    // }
    ICPExample().Run();

    return 0;
}
