# coding: utf-8

import os
import os.path as osp
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# User import
from configs import cfg as gcfg  # initialization


def custom_draw_geometry_with_camera_trajectory(pcd):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
            o3d.io.read_pinhole_camera_trajectory(
                    "../examples/TestData/camera_trajectory.json")
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    if not os.path.exists("../examples/TestData/image/"):
        os.makedirs("../examples/TestData/image/")
    if not os.path.exists("../examples/TestData/depth/"):
        os.makedirs("../examples/TestData/depth/")

    save_dir = osp.join(gcfg.get_ou_dir, 'image-seq')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave("../examples/TestData/depth/{:05d}.png".format(glb.index),\
                    np.asarray(depth), dpi = 1)
            plt.imsave("../examples/TestData/image/{:05d}.png".format(glb.index),\
                    np.asarray(image), dpi = 1)
            #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                    register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("../examples/TestData/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    xyzrgb_path = '/disk4t0/HOME_perception_module/MonoDepth-Home/output/' \
                  '20190924_132946_435313_0rgb_estimated.xyzrgb'
    print('Load point cloud')
    xyzrgb = o3d.io.read_point_cloud(xyzrgb_path)
    # o3d.visualization.draw_geometries([xyzrgb])

    custom_draw_geometry_with_camera_trajectory(xyzrgb)
    pass
