#!/usr/bin/env python

import glob
import numpy as np
import os
import rosbag
import ros_numpy
import rospy
import sys
import tf
import yaml
from transforms3d.quaternions import *
from transforms3d.axangles import axangle2mat


def align_cameras(data_dir, update=False):
    cam1_aligned_pose = [0.00022101, 0.014945, -3.8073e-05, 0.999999, 0.00057514, 0.00054611, -0.00082246]
    cam2_aligned_pose = [0.0002763, 0.014992, 5.3916e-05, 0.999994, 0.00024522, -0.0015323, -0.0031606]
    cam3_aligned_pose = [0.0001781, 0.014865, 9.6475e-05, 0.999992, 0.00044859, -0.0023165, -0.0033206]
    cam4_aligned_pose = [-0.00019038, 0.014772, 5.346e-05, 0.999999, 0.00033169, -0.00072087, 0.00073762]
    depth_optical_frame_pose = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]

    with open(os.path.join(data_dir, 'camera_pose_world.yml'), 'r') as f:
        camera_pose_dict = yaml.load(f, Loader=yaml.FullLoader)

    for i, pose in enumerate([cam1_aligned_pose, cam2_aligned_pose, cam3_aligned_pose, cam4_aligned_pose]):
        cam_ori = np.array(camera_pose_dict[f"cam_{i+1}"]["orientation"])
        cam_ori_new = qmult(qinverse(pose[3:]), cam_ori)
        cam_pos = np.array(camera_pose_dict[f"cam_{i+1}"]["position"])
        # cam_pos_new = (quat2mat(depth_optical_frame_pose[3:]) @ cam_pos.T).T + depth_optical_frame_pose[:3]
        cam_pos_new = cam_pos - (quat2mat(cam_ori_new) @ np.array(pose[:3]).T).T

        if update:
            camera_pose_dict[f"cam_{i+1}"]["orientation"] = cam_ori_new.tolist()
            camera_pose_dict[f"cam_{i+1}"]["position"] = cam_pos_new.tolist()

    if update:
        with open(os.path.join(data_dir, 'camera_pose_world_aligned.yml'), 'w') as f:
            yaml.dump(camera_pose_dict, f)


def main():
    # rospy.init_node('cam_pose_transformer', anonymous=True)

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_dir = os.path.join(cd, '..', 'env')

    align_cameras(data_dir, update=True)


if __name__ == '__main__':
    main()
