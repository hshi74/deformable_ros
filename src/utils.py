
import numpy as np
import open3d as o3d
import os
import ros_numpy
import sys
import yaml

from act_from_param import *
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import *

cd = os.path.dirname(os.path.realpath(sys.argv[0]))
with open(os.path.join(cd, '..', 'env', 'camera_pose_world.yml'), 'r') as f:
    cam_pose_dict = yaml.load(f, Loader=yaml.FullLoader)

depth_optical_frame_pose = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]


def get_center(pcd_msgs, visualize=False):
    pcd_all = o3d.geometry.PointCloud()
    for i in range(len(pcd_msgs)):
        cloud_rec = ros_numpy.point_cloud2.pointcloud2_to_array(pcd_msgs[i])
        cloud_array = cloud_rec.view('<f4').reshape(cloud_rec.shape + (-1,))
        points = cloud_array[:, :3]
        points = (quat2mat(depth_optical_frame_pose[3:]) @ points.T).T + depth_optical_frame_pose[:3]
        cam_ori = cam_pose_dict[f"cam_{i+1}"]["orientation"]
        points = (quat2mat(cam_ori) @ points.T).T + cam_pose_dict[f"cam_{i+1}"]["position"]
        
        cloud_rgb_bytes = cloud_array[:, -1].tobytes()
        # int.from_bytes(cloud_rgb_bytes, 'big')
        cloud_bgr = np.frombuffer(cloud_rgb_bytes, dtype=np.uint8).reshape(-1, 4) / 255 
        cloud_rgb = cloud_bgr[:, ::-1]

        x_filter = (points.T[0] > 0.4 - 0.1) & (points.T[0] < 0.4 + 0.1)
        y_filter = (points.T[1] > -0.1 - 0.1) & (points.T[1] < -0.1 + 0.1)
        z_filter = (points.T[2] > 0 + 0.005) & (points.T[2] < 0 + 0.07)
        points = points[x_filter & y_filter & z_filter]
        cloud_rgb = cloud_rgb[x_filter & y_filter & z_filter, 1:]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(cloud_rgb)

        pcd_all += pcd

    if visualize:
        o3d.visualization.draw_geometries([pcd_all])

    pcd_colors = np.asarray(pcd_all.colors)
    # color filters
    cube_label = np.where(np.logical_and(pcd_colors[:, 0] < 0.2, pcd_colors[:, 2] > 0.2))
    cube = pcd_all.select_by_index(cube_label[0])

    cube = cube.voxel_down_sample(voxel_size=0.001)

    cl, inlier_ind_cube_stat = cube.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
    cube = cube.select_by_index(inlier_ind_cube_stat)
    
    if visualize:
        o3d.visualization.draw_geometries([cube])

    mid_point = list(np.mean(np.asarray(cube.points), axis=0))

    return mid_point
