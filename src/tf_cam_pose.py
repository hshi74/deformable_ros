#!/usr/bin/env python

import glob
import numpy as np
import open3d as o3d
import os
import rosbag
import ros_numpy
import rospy
import sys
import tf
import yaml
from transforms3d.quaternions import *
from transforms3d.axangles import axangle2mat


def ar_tag_transform(data_dir, num_cams=4):
    with open(os.path.join(data_dir, 'robot_state.yml'), 'r') as f:
        robot_state_dict = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(data_dir, 'camera_tag.yml'), 'r') as f:
        camera_tag_dict = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(data_dir, 'camera_pose.yml'), 'r') as f:
        camera_pose_dict = yaml.load(f, Loader=yaml.FullLoader)

    # tf_listener = tf.TransformListener()
    # tf_listener.waitForTransform(f"cam1_link", f'cam1_color_optical_frame', rospy.Time.now(), rospy.Duration(1.0))
    # cam_frame_trans, cam_frame_quat = tf_listener.lookupTransform( f"cam1_link", f'cam1_color_optical_frame', rospy.Time.now())
    
    # cam_frame_rot = quat2mat([cam_frame_quat[3], cam_frame_quat[0], cam_frame_quat[1], cam_frame_quat[2]])
    # T_cam_frame = np.concatenate((np.concatenate((cam_frame_rot, np.array([cam_frame_trans]).T), axis=1), [[0, 0, 0, 1]]))

    # transformation of tag in the camera frame
    cam_rot = quat2mat(camera_tag_dict['cam_1']['orientation'])
    T_cam_tag = np.concatenate((np.concatenate((cam_rot, np.array([camera_tag_dict['cam_1']['position']]).T), axis=1), [[0, 0, 0, 1]]))
    # transformation of camera link in the tag frame
    T_tag_frame = np.linalg.inv(T_cam_tag)

    ar_tag_rot = quat2mat(axangle2quat([0, 0, 1], np.pi / 2))
    T_world_tag = np.concatenate((np.concatenate((ar_tag_rot, np.array([robot_state_dict['fingertip_pos']]).T), axis=1), [[0, 0, 0, 1]]))
    T_world_frame = T_world_tag @ T_tag_frame

    for i in range(1, num_cams + 1):
        new_rot = qmult(mat2quat(T_world_frame[:3, :3]), camera_pose_dict[f"cam_{i}"]["orientation"])
        camera_pose_dict[f"cam_{i}"]["orientation"] = new_rot.tolist()
        new_pos = T_world_frame[:3, :3] @ np.array(camera_pose_dict[f"cam_{i}"]["position"]) + T_world_frame[:3, 3]
        camera_pose_dict[f"cam_{i}"]["position"] = new_pos.tolist()

    with open(os.path.join(data_dir, 'camera_pose_world.yml'), 'w') as f:
        yaml.dump(camera_pose_dict, f)


def align_plane_transform(data_dir, pcd_msgs, update=False):
    depth_optical_frame_pose = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]

    with open(os.path.join(data_dir, 'camera_pose_align.yml'), 'r') as f:
        camera_pose_dict = yaml.load(f, Loader=yaml.FullLoader)

    for i in range(len(pcd_msgs)):
        cloud_rec = ros_numpy.point_cloud2.pointcloud2_to_array(pcd_msgs[i])
        cloud_array = cloud_rec.view('<f4').reshape(cloud_rec.shape + (-1,))
        points = cloud_array[:, :3]
        points = (quat2mat(depth_optical_frame_pose[3:]) @ points.T).T + depth_optical_frame_pose[:3]
        cam_ori = camera_pose_dict[f"cam_{i+1}"]["orientation"]
        points = (quat2mat(cam_ori) @ points.T).T + camera_pose_dict[f"cam_{i+1}"]["position"]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        segment_models, inliers = pcd.segment_plane(distance_threshold=0.0075,ransac_n=3,num_iterations=100)
        [a, b, c, d] = segment_models
        print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        if update:
            align_mat = axangle2mat([b, -a, 0], np.arccos(c / np.linalg.norm([a, b, c])))

            new_rot = qmult(mat2quat(align_mat), cam_ori)
            camera_pose_dict[f"cam_{i+1}"]["orientation"] = new_rot.tolist()
            new_pos = align_mat @ np.array(camera_pose_dict[f"cam_{i+1}"]["position"]) + [0, 0, d / c]
            camera_pose_dict[f"cam_{i+1}"]["position"] = new_pos.tolist()

    if update:
        with open(os.path.join(data_dir, 'camera_pose_align.yml'), 'w') as f:
            yaml.dump(camera_pose_dict, f)


def main():
    # rospy.init_node('cam_pose_transformer', anonymous=True)

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_dir = os.path.join(cd, '..', 'env')

    # ar_tag_transform(data_dir)
    
    bag_path = os.path.join(data_dir, '..', 'raw_data', 'calibration', 'plasticine_0.bag')

    bag = rosbag.Bag(bag_path)

    pcd_msgs = []
    for topic, msg, t in bag.read_messages(
        topics=['/cam1/depth/color/points', '/cam2/depth/color/points', '/cam3/depth/color/points', '/cam4/depth/color/points']
    ):
        pcd_msgs.append(msg)
    bag.close()

    align_plane_transform(data_dir, pcd_msgs, update=True)
    align_plane_transform(data_dir, pcd_msgs, update=False)


if __name__ == '__main__':
    main()
