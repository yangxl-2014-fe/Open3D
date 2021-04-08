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

// This is an example for Multi-Scale ICP Registration.
// This takes a config.txt file, an example of which is provided in
// Open3D/examples/test_data/ICP/TMultiScaleICPConfig.txt.
//
// Command to run this from Open3D build directory:
// ./bin/examples/TICPRegistration [Device] [Path to Config]
// [Device]: CPU:0 / CUDA:0 ...
// [Sample Config Path]: ../examples/test_data/ICP/TMultiScaleICPConfig.txt

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

const std::string path_source =
        "../../../examples/test_data/ICP/cloud_bin_0.pcd";
const std::string path_target =
        "../../../examples/test_data/ICP/cloud_bin_1.pcd";

class ExampleWindow {
public:
    ExampleWindow(const core::Device& device)
        : device_(device), dtype_(core::Dtype::Float32) {
        is_done_ = false;
        visualization::gui::Application::GetInstance().Initialize();
    }

    void Run() {
        main_vis_ = std::make_shared<visualization::visualizer::O3DVisualizer>(
                "Open3D - Multi-Scale ICP Demo", WIDTH, HEIGHT);
        main_vis_->AddAction("Run MultiScale ICP",
                             [this](visualization::visualizer::O3DVisualizer&) {
                                 this->RunMultiScaleICP();
                             });
        main_vis_->SetOnClose([this]() { return this->OnMainWindowClosing(); });
        visualization::gui::Application::GetInstance().AddWindow(main_vis_);
        std::thread read_thread([this]() { this->UpdateThreadMain(); });
        visualization::gui::Application::GetInstance().Run();
        read_thread.join();
    }

private:
    void RunMultiScaleICP() {
        std::cout << " Test 9" << std::endl;
        return;
    }

    bool OnMainWindowClosing() {
        // Ensure object is free so Filament can clean up without crashing.
        // Also signals to the "reading" thread that it is finished.
        main_vis_.reset();
        return true;  // false would cancel the close
    }

    void UpdateThreadMain() {
        // Load data.
        // Pre-process data.

        // Using Legacy PointCloud for visualization.
        {
            std::lock_guard<std::mutex> lock(cloud_lock_);

            source_ = std::make_shared<geometry::PointCloud>();
            io::ReadPointCloud(path_source, *source_);

            target_ = std::make_shared<geometry::PointCloud>();
            io::ReadPointCloud(path_target, *target_);
        }

        geometry::AxisAlignedBoundingBox bounds;
        Eigen::Vector3d extent;
        {
            std::lock_guard<std::mutex> lock(cloud_lock_);

            bounds = source_->GetAxisAlignedBoundingBox();
            extent = bounds.GetExtent();
        }

        // Using Tensor PointCloud for processing.
        // auto tsource =
        // t::geometry::PointCloud::FromLegacyPointCloud(*source_); auto ttarget
        // = t::geometry::PointCloud::FromLegacyPointCloud(*target_);

        // Load PointCloud as Float32 on device, for processing.
        // ProcessAndLoadPointCloud(tsource);
        // ProcessAndLoadPointCloud(ttarget);

        core::Tensor transformation_local =
                core::Tensor::Eye(4, dtype_, device_);

        auto eigen = core::eigen_converter::TensorToEigenMatrixXd(
                transformation_local);
        std::cout << " Eigen: " << eigen << std::endl;

        source_->Transform(eigen);

        auto mat = visualization::rendering::Material();
        mat.shader = "defaultUnlit";

        visualization::gui::Application::GetInstance().PostToMainThread(
                main_vis_.get(), [this, bounds, mat]() {
                    std::lock_guard<std::mutex> lock(cloud_lock_);
                    main_vis_->AddGeometry(SRC_CLOUD, source_, &mat);
                    main_vis_->AddGeometry(DST_CLOUD, target_, &mat);
                    main_vis_->ResetCameraToDefault();
                    Eigen::Vector3f center = bounds.GetCenter().cast<float>();
                    main_vis_->SetupCamera(60, center, center + CENTER_OFFSET,
                                           {0.0f, -1.0f, 0.0f});
                });

        // bool is_initialized = false;
        while (!is_done_) {
            // std::stringstream out;
            // DoSingleICPIterarion.
            // Get Transformation tensor.
            // Get Eigen Matrix4d from transformation tensor.
            // Update Source PointCloud.

            // Visualize the pointclouds.
            visualization::gui::Application::GetInstance().PostToMainThread(
                    main_vis_.get(), [this, mat]() {
                        // this->SetOutput(out);
                        std::lock_guard<std::mutex> lock(cloud_lock_);

                        main_vis_->RemoveGeometry(SRC_CLOUD);

                        main_vis_->AddGeometry(SRC_CLOUD, source_, &mat);
                    });

            // is_initialized = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // void ProcessAndLoadPointCloud(t::geometry::PointCloud& pcd) {
    //     pcd.SetPoints(pcd.GetPoints().To(device_, dtype_, /*copy=*/true));
    //     if (pcd.HasPointNormals()) {
    //         pcd.SetPointNormals(
    //                 pcd.GetPointNormals().To(device_, dtype_,
    //                 /*copy=*/true));
    //     }
    //     if (pcd.HasPointColors()) {
    //         pcd.SetPointColors(
    //                 pcd.GetPointColors().To(device_, dtype_, /*copy=*/true));
    //     }
    // }

private:
    core::Device device_;
    core::Dtype dtype_;

    std::mutex cloud_lock_;
    // Source PointCloud on CPU, used for visualization.
    std::shared_ptr<geometry::PointCloud> source_;
    // Target PointCloud on CPU, used for visualization.
    std::shared_ptr<geometry::PointCloud> target_;
    // // Source PointCloud on Device, used for visualization.
    // std::shared_ptr<t::geometry::PointCloud> source_device;
    // // Target PointCloud on Device, used for visualization.
    // std::shared_ptr<t::geometry::PointCloud> target_device_;

    std::shared_ptr<visualization::gui::SceneWidget> widget3d_;

    std::atomic<bool> is_done_;
    std::shared_ptr<visualization::visualizer::O3DVisualizer> main_vis_;
};

//------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
    // if (!utility::filesystem::DirectoryExists(path_source)) {
    //     utility::LogError(
    //             "This example needs to be run from the <build> "
    //             "directory");
    // }
    core::Device device("CPU:0");
    ExampleWindow(device).Run();
    return 0;
}
