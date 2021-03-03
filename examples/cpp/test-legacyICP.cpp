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

// This example tests ICP Registration pipeline on the given pointcloud.
// To make things simple, and support any pointcloud for testing, input only
// requires 1 pointcloud source in argument, and the example automatically
// creates a target source by transforming the pointcloud, and estimating
// normals. Adjust the voxel_downsample_factor and max_correspondence_dist
// according to the test pointcloud.
//
//
// To run this example from Open3D directory:
// ./build/bin/example/test-tICP [device] [path to source pointcloud]
// [device] : CPU:0 / CUDA:0 ...
// [example path to source pointcloud relative to Open3D dir]:
// examples/test_data/ICP/cloud_bin_0.pcd

#include <iostream>
#include <memory>

#include "open3d/Open3D.h"

using namespace open3d;

// Parameters to adjust according to the test pointcloud.
double voxel_downsample_factor = 0.5;
double max_correspondence_dist = 3.0;

// ICP ConvergenceCriteria:
double relative_fitness = 1e-6;
double relative_rmse = 1e-6;
int max_iterations = 5;

void VisualizeRegistration(const open3d::geometry::PointCloud &source,
                           const open3d::geometry::PointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    std::shared_ptr<geometry::PointCloud> source_transformed_ptr(
            new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> target_ptr(new geometry::PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(Transformation);
    visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  "Registration result");
}

int main(int argc, char *argv[]) {
    // Argument 1: Path to the source PointCloud
    // Argument 2: Path to the target PointCloud

    // Creating Tensor PointCloud Input from argument specified file
    // std::shared_ptr<open3d::geometry::PointCloud> source =
    //         open3d::io::CreatePointCloudFromFile(argv[1]);
    std::shared_ptr<open3d::geometry::PointCloud> target_ =
            open3d::io::CreatePointCloudFromFile(argv[1]);

    utility::LogInfo(" Input Successful ");
    geometry::PointCloud legacy_t = *target_;
    legacy_t = *legacy_t.VoxelDownSample(voxel_downsample_factor);
    utility::LogInfo(" Downsampling Successful ");

    legacy_t.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(), false);
    utility::LogInfo(" Normal Estimation Successful ");

    geometry::PointCloud source = legacy_t;
    geometry::PointCloud target = legacy_t;

    Eigen::Matrix4d trans;
    trans << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7, 0.487,
            0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;
    target.Transform(trans);
    utility::LogInfo(" Target transformation Successful ");

    Eigen::Matrix4d init_trans = Eigen::Matrix4d::Identity();

    utility::LogInfo(" Input Process on Legacy CPU Success");

    VisualizeRegistration(source, target, init_trans);

    open3d::pipelines::registration::RegistrationResult evaluation(init_trans);

    evaluation = open3d::pipelines::registration::EvaluateRegistration(
            source, target, max_correspondence_dist, init_trans);
    utility::LogInfo(" EvaluateRegistration Success");

    // ICP: Point to Plane
    utility::Timer icp_p2plane_time;
    icp_p2plane_time.Start();
    auto reg_p2plane = open3d::pipelines::registration::RegistrationICP(
            source, target, max_correspondence_dist, init_trans,
            open3d::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    icp_p2plane_time.Stop();

    // Printing result for ICP Point to Plane
    utility::LogInfo(" [ICP: Point to Plane] ");
    utility::LogInfo(
            "   EvaluateRegistration on [Legacy CPU Implementation] Success ");
    utility::LogInfo("   Convergence Criteria: ");
    utility::LogInfo(
            "   Relative Fitness: {}, Relative Fitness: {}, Max Iterations {}",
            relative_fitness, relative_rmse, max_iterations);
    utility::LogInfo(
            "   [Correspondences]: {}, [maximum corrspondence distance = {}] ",
            reg_p2plane.correspondence_set_.size(), max_correspondence_dist);
    utility::LogInfo("     Fitness: {} ", reg_p2plane.fitness_);
    utility::LogInfo("     Inlier RMSE: {} ", reg_p2plane.inlier_rmse_);
    utility::LogInfo("     [Time]: {}", icp_p2plane_time.GetDuration());
    utility::LogInfo("     [Tranformation Matrix]: ");
    std::cout << reg_p2plane.transformation_ << std::endl;

    VisualizeRegistration(source, target, reg_p2plane.transformation_);

    return 0;
}
