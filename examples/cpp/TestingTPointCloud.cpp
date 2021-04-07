// Experiment.

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::t::pipelines::registration;

std::string path_source = "../examples/test_data/ICP/cloud_bin_0.pcd";
core::Device device("CPU:0");
core::Dtype dtype = core::Dtype::Float32;

int main(int argc, char** argv) {
    t::geometry::PointCloud pcd(device);
    auto pcd_ptr = std::make_shared<t::geometry::PointCloud>(device);

    // t::io::ReadPointCloud copies the pointcloud to CPU.
    t::io::ReadPointCloud(path_source, pcd, {"auto", false, false, true});

    for (std::string attr : {"points", "colors", "normals"}) {
        if (pcd.HasPointAttr(attr)) {
            pcd_ptr->SetPointAttr(attr, pcd.GetPointAttr(attr).To(
                                                device, dtype, /*copy=*/true));
        }
    }

    visualization::DrawGeometries({pcd_ptr});

    return 0;
}