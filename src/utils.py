
import cv2 as cv
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


def get_cube_center(pcd_msgs, visualize=False):
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

        # xyz filter
        x_filter = (points.T[0] > 0.4 - 0.05) & (points.T[0] < 0.4 + 0.1)
        y_filter = (points.T[1] > -0.1 - 0.05) & (points.T[1] < -0.1 + 0.1)
        z_filter = (points.T[2] > 0 + 0.002) & (points.T[2] < 0 + 0.07) # or 0.005
        points = points[x_filter & y_filter & z_filter]
        cloud_rgb = cloud_rgb[x_filter & y_filter & z_filter, 1:]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(cloud_rgb)

        pcd_all += pcd

    if visualize:
        o3d.visualization.draw_geometries([pcd_all])
    
    # color filter
    pcd_colors = np.asarray(pcd_all.colors, dtype=np.float32)
    # bgr
    pcd_rgb = pcd_colors[None, :, :]

    pcd_hsv = cv.cvtColor(pcd_rgb, cv.COLOR_RGB2HSV)
    hsv_lower = np.array([0, 0, 0])
    hsv_upper = np.array([120, 255, 255])
    mask = cv.inRange(pcd_hsv, hsv_lower, hsv_upper)
    cube_label = np.where(mask[0] == 255)
    
    cube = pcd_all.select_by_index(cube_label[0])
    rest = pcd_all.select_by_index(cube_label[0], invert=True)

    cube = cube.voxel_down_sample(voxel_size=0.001)

    cl, inlier_ind_cube_stat = cube.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.5)
    cube = cube.select_by_index(inlier_ind_cube_stat)
    
    if visualize:
        o3d.visualization.draw_geometries([cube])

    if visualize:
        o3d.visualization.draw_geometries([rest])

    # import pdb; pdb.set_trace()

    cube_points = np.asarray(cube.points)
    cube_proj_points = np.concatenate((cube_points[:, :2], np.zeros((cube_points.shape[0], 1))), axis=1)
    cube_proj_pcd = o3d.geometry.PointCloud()
    cube_proj_pcd.points = o3d.utility.Vector3dVector(cube_proj_points)
    cube_proj_pcd = cube_proj_pcd.voxel_down_sample(voxel_size=0.003)

    if visualize:
        o3d.visualization.draw_geometries([cube_proj_pcd])

    return cube, cube_proj_pcd.get_center()[:2]
