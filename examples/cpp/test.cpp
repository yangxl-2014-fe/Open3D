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

void MultiScaleICP() {
    // Take input as Legacy PointCloud.
    auto source = std::make_shared<geometry::PointCloud>();
    io::ReadPointCloud(path_src, *source);
    if (source->points_.empty()) {
        utility::LogError("Could not open {}", SRC_CLOUD);
        return;
    }
    auto target = std::make_shared<geometry::PointCloud>();
    io::ReadPointCloud(path_dst, *target);
    if (target->points_.empty()) {
        utility::LogError("Could not open {}", DST_CLOUD);
        return;
    }
    std::cout << " PointCloud uploaded " << std::endl;
    // source->PaintUniformColor({1.000, 0.706, 0.000});
    // target->PaintUniformColor({0.000, 0.651, 0.929});

    // Convert to Tensor PointCloud for processing.

    t::geometry::PointCloud source_t =
            t::geometry::PointCloud::FromLegacyPointCloud(*source)
                    .VoxelDownSample(0.05);
    t::geometry::PointCloud target_t =
            t::geometry::PointCloud::FromLegacyPointCloud(*target)
                    .VoxelDownSample(0.05);

    core::Tensor transformation_device =
            core::Tensor::Eye(4, core::Dtype::Float32, core::Device("CPU:0"));

    auto DoSingleIterationICP =
            [source, target, source_t, target_t, 
              transformation_device](visualization::visualizer::O3DVisualizer& o3dvis) {
                std::cout << " Inside Action " << std::endl;
                auto result = RegistrationICP(
                        source_t, target_t, 0.02, transformation_device,
                        TransformationEstimationPointToPlane(),
                        ICPConvergenceCriteria());
                transformation_device =
                        (result.transformation_).Matmul(transformation_device);
                auto t = core::eigen_converter::TensorToEigenMatrixXd(
                        result.transformation_);
                std::cout << " Transformation: " << t << std::endl;
                source->Transform(t);
                std::cout << " Transformation done " << std::endl;

                // Update the source geometry
                o3dvis.RemoveGeometry(SRC_CLOUD);
                std::cout << " Geometry removed " << std::endl;
                o3dvis.AddGeometry(SRC_CLOUD, source);
                std::cout << " Geometry added " << std::endl;
            };

    visualization::Draw({visualization::DrawObject(SRC_CLOUD, source),
                         visualization::DrawObject(DST_CLOUD, target)},
                        "Open3D: ICP Registration example: Selection", 1024,
                        768, {{"ICP Registration ", DoSingleIterationICP}});
}

int main(int argc, const char* argv[]) {
    // if (!utility::filesystem::DirectoryExists(path_source)) {
    //     utility::LogError(
    //             "This example needs to be run from the <build> "
    //             "directory");
    // }
    MultiScaleICP();

    return 0;
}
