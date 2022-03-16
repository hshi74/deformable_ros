#!/usr/bin/env python

import math
from turtle import position
import numpy as np
import os
import rospy
import sys
import tf
import torch
import yaml
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import Pose
from tqdm import tqdm
from transforms3d.quaternions import *


def main():
    rospy.init_node('cam_pose_transformer', anonymous=True)

    num_cams = 4

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_dir = os.path.join(cd, '..', 'env')
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

if __name__ == '__main__':
    main()
