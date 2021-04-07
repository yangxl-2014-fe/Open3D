// Demo Experiment for Multi Scale ICP.

#include <fstream>
#include <sstream>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::t::pipelines::registration;

// For each frame registration using MultiScaleICP.
std::vector<double> voxel_sizes;
std::vector<double> search_radius;
std::vector<ICPConvergenceCriteria> criterias;

std::string path_config_file;
std::string path_source;
std::string path_target;
std::string registration_method;
std::string verbosity;

core::Device device;
core::Dtype dtype;
core::Tensor initial_transformation;

void PrintHelp() {
    PrintOpen3DVersion();
    utility::LogInfo("Usage :");
    utility::LogInfo("    > TMultiScaleICP [device] [path to config.txt file]");
}

void ReadConfigFile();

void CopyPointCloud(std::shared_ptr<t::geometry::PointCloud> pcd_ptr,
                    t::geometry::PointCloud& pcd,
                    core::Dtype dtype,
                    core::Device device);

void LoadTensorPointClouds(std::shared_ptr<t::geometry::PointCloud> source_ptr,
            std::shared_ptr<t::geometry::PointCloud> target_ptr, core::Dtype dtype, core::Device device);

void VisualizeRegistration() {
    auto source_ptr = std::make_shared<t::geometry::PointCloud>(device);
    auto target_ptr = std::make_shared<t::geometry::PointCloud>(device);
    LoadTensorPointClouds(source_ptr, target_ptr, dtype, device);

    std::cout << " source: " << source_ptr->GetPoints().GetShape().ToString()
              << std::endl;
    std::cout << " target: " << target_ptr->GetPoints().GetShape().ToString()
              << std::endl;

    // const char *source_name = "Source (yellow)";
    // const char *target_name = "Target (blue)";

    // t::pipelines::registration::RegistrationResult result(initial_transformation);

    // std::shared_ptr<TransformationEstimation> estimation;
    // if (registration_method == "PointToPoint") {
    //     estimation =
    //     std::make_shared<TransformationEstimationPointToPoint>();
    // } else if (registration_method == "PointToPlane") {
    //     estimation =
    //     std::make_shared<TransformationEstimationPointToPlane>();
    // } else {
    //     utility::LogError(" Registration method {}, not implemented.",
    //                       registration_method);
    // }

    // // Warm Up.
    // result = RegistrationICPMultiScale(
    //         *source_ptr, *source_ptr, {1.0},  {ICPConvergenceCriteria(0.01,
    //         0.01, 1)}, {1.5}, core::Tensor::Eye(4, dtype, device),
    //         *estimation);

    // std::cout << " hey 2 " << std::endl;
    // auto RunMultiScaleICP =
    //         [source_ptr, target_ptr, estimation, source_name,
    //          target_name](visualization::visualizer::O3DVisualizer &o3dvis) {
    //             t::pipelines::registration::RegistrationResult result(
    //                     initial_transformation);
    //             result = RegistrationICPMultiScale(*source_ptr, *target_ptr,
    //                             voxel_sizes, criterias,
    //                             search_radius, initial_transformation,
    //                             *estimation);
    //             source_ptr->Transform(result.transformation_);

    //             // Update the source geometry
    //             o3dvis.RemoveGeometry(source_name);
    //             o3dvis.AddGeometry(source_name, source_ptr);
    //         };
    std::cout << " hey 2 " << std::endl;
    /*
        visualization::Draw({visualization::DrawObject(source_name, source_ptr),
                             visualization::DrawObject(target_name,
       target_ptr)}, "Open3D: Draw example: Selection", 1024, 768,
                            {{"Run Multi Scale ICP", RunMultiScaleICP}});*/
}

int main(int argc, char** argv) {
    if (argc == 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        argc != 3) {
        PrintHelp();
        return 1;
    }
    // Verbosity can be changes in the config file.
    utility::VerbosityLevel verb;
    if (verbosity == "Debug") {
        verb = utility::VerbosityLevel::Debug;
    } else {
        verb = utility::VerbosityLevel::Info;
    }
    utility::SetVerbosityLevel(verb);

    device = core::Device(argv[1]);
    dtype = core::Dtype::Float32;
    path_config_file = std::string(argv[2]);

    ReadConfigFile();

    std::vector<float> initial_transform_flat = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                 0.0, 0.0, 0.0, 1.0};

    initial_transformation =
            core::Tensor(initial_transform_flat, {4, 4}, dtype, device);

    VisualizeRegistration();

    return 0;
}

void CopyPointCloud(std::shared_ptr<t::geometry::PointCloud> pcd_ptr,
                    t::geometry::PointCloud& pcd,
                    core::Dtype dtype,
                    core::Device device) {
    for (std::string attr : {"points", "colors", "normals"}) {
        if (pcd.HasPointAttr(attr)) {
            pcd_ptr->SetPointAttr(attr, pcd.GetPointAttr(attr).To(
                                                device, dtype, /*copy=*/true));
        }
    }
}

// std::tuple<std::shared_ptr<t::geometry::PointCloud>,
//            std::shared_ptr<t::geometry::PointCloud>>
void LoadTensorPointClouds(std::shared_ptr<t::geometry::PointCloud> source_ptr,
            std::shared_ptr<t::geometry::PointCloud> target_ptr, core::Dtype dtype, core::Device device) {
    t::geometry::PointCloud source, target;

    // t::io::ReadPointCloud copies the pointcloud to CPU.
    t::io::ReadPointCloud(path_source, source, {"auto", false, false, true});
    t::io::ReadPointCloud(path_target, target, {"auto", false, false, true});

    // Currently only Float32 pointcloud is supported.
    source = source.To(device);
    target = target.To(device);

    if (registration_method == "PointToPlane" && !target.HasPointNormals()) {
        auto target_legacy = target.ToLegacyPointCloud();
        target_legacy.EstimateNormals(geometry::KDTreeSearchParamKNN(), false);
        core::Tensor target_normals =
                t::geometry::PointCloud::FromLegacyPointCloud(target_legacy)
                        .GetPointNormals()
                        .To(device, dtype);
        target.SetPointNormals(target_normals);
    }

    CopyPointCloud(source_ptr, source, dtype, device);
    CopyPointCloud(target_ptr, target, dtype, device);
}

void ReadConfigFile() {
    std::ifstream cFile(path_config_file);
    std::vector<double> relative_fitness;
    std::vector<double> relative_rmse;
    std::vector<int> max_iterations;

    if (cFile.is_open()) {
        std::string line;
        while (getline(cFile, line)) {
            line.erase(std::remove_if(line.begin(), line.end(), isspace),
                       line.end());
            if (line[0] == '#' || line.empty()) continue;

            auto delimiterPos = line.find("=");
            auto name = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);

            if (name == "source_path") {
                path_source = value;
            } else if (name == "target_path") {
                path_target = value;
            } else if (name == "registration_method") {
                registration_method = value;
            } else if (name == "criteria.relative_fitness") {
                std::istringstream is(value);
                relative_fitness.push_back(std::stod(value));
            } else if (name == "criteria.relative_rmse") {
                std::istringstream is(value);
                relative_rmse.push_back(std::stod(value));
            } else if (name == "criteria.max_iterations") {
                std::istringstream is(value);
                max_iterations.push_back(std::stoi(value));
            } else if (name == "voxel_size") {
                std::istringstream is(value);
                voxel_sizes.push_back(std::stod(value));
            } else if (name == "search_radii") {
                std::istringstream is(value);
                search_radius.push_back(std::stod(value));
            } else if (name == "verbosity") {
                std::istringstream is(value);
                verbosity = value;
            }
        }
    } else {
        std::cerr << "Couldn't open config file for reading.\n";
    }

    utility::LogInfo(" Source path: {}", path_source);
    utility::LogInfo(" Target path: {}", path_target);
    utility::LogInfo(" Registrtion method: {}", registration_method);
    std::cout << std::endl;

    std::cout << " Voxel Sizes: ";
    for (auto voxel_size : voxel_sizes) std::cout << voxel_size << " ";
    std::cout << std::endl;

    std::cout << " Search Radius Sizes: ";
    for (auto search_radii : search_radius) std::cout << search_radii << " ";
    std::cout << std::endl;

    std::cout << " ICPCriteria: " << std::endl;
    std::cout << "   Max Iterations: ";
    for (auto iteration : max_iterations) std::cout << iteration << " ";
    std::cout << std::endl;
    std::cout << "   Relative Fitness: ";
    for (auto fitness : relative_fitness) std::cout << fitness << " ";
    std::cout << std::endl;
    std::cout << "   Relative RMSE: ";
    for (auto rmse : relative_rmse) std::cout << rmse << " ";
    std::cout << std::endl;

    size_t length = voxel_sizes.size();
    if (search_radius.size() != length || max_iterations.size() != length ||
        relative_fitness.size() != length || relative_rmse.size() != length) {
        utility::LogError(
                " Length of vector: voxel_sizes, search_sizes, max_iterations, "
                "relative_fitness, relative_rmse must be same.");
    }

    for (int i = 0; i < (int)length; i++) {
        auto criteria = ICPConvergenceCriteria(
                relative_fitness[i], relative_rmse[i], max_iterations[i]);
        criterias.push_back(criteria);
    }

    std::cout << " Press Enter To Continue... " << std::endl;
    std::getchar();
}
