import time
import numpy as np
import os

import open3d as o3d
import readchar
import rosbag
import rospy
import ros_numpy
import sys
import tf
import tf2_ros
import yaml

from datetime import datetime
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from timeit import default_timer as timer
from transforms3d.quaternions import *

fixed_frame = 'panda_link0'
num_cams = 4
cd = os.path.dirname(os.path.realpath(sys.argv[0]))
data_dir = os.path.join(cd, '..', 'env')
with open(os.path.join(data_dir, 'camera_pose_world.yml'), 'r') as f:
    camera_pose_dict = yaml.load(f, Loader=yaml.FullLoader)

pos_stride = 0.0005
rot_stride = 0.001

def main():
    rospy.init_node('cam_pose_tuner', anonymous=True)

    tune_idx = 4
    static_br = tf2_ros.StaticTransformBroadcaster()
    static_ts_list = []
    for i in [1, 2, 3, 4]:
        static_ts = TransformStamped()
        # static_ts.header.stamp = rospy.Time.now()
        static_ts.header.frame_id = fixed_frame
        static_ts.child_frame_id = f"cam{i}_link"

        static_ts.transform.translation.x = camera_pose_dict[f"cam_{i}"]["position"][0]
        static_ts.transform.translation.y = camera_pose_dict[f"cam_{i}"]["position"][1]
        static_ts.transform.translation.z = camera_pose_dict[f"cam_{i}"]["position"][2]

        static_ts.transform.rotation.x = camera_pose_dict[f"cam_{i}"]["orientation"][1]
        static_ts.transform.rotation.y = camera_pose_dict[f"cam_{i}"]["orientation"][2]
        static_ts.transform.rotation.z = camera_pose_dict[f"cam_{i}"]["orientation"][3]
        static_ts.transform.rotation.w = camera_pose_dict[f"cam_{i}"]["orientation"][0]
        
        static_ts_list.append(static_ts)

    static_br.sendTransform(static_ts_list)

    cam_idx = tune_idx
    cam_pos_world = camera_pose_dict[f"cam_{cam_idx}"]["position"]
    cam_ori_world_init = camera_pose_dict[f"cam_{cam_idx}"]["orientation"]
    pcd_rot_vec = [0.0, 0.0, 0.0]
    
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        key = readchar.readkey()
        if key == 'w':
            cam_pos_world[0] += pos_stride
        elif key == 'x':
            cam_pos_world[0] -= pos_stride
        elif key == 'a':
            cam_pos_world[1] += pos_stride
        elif key == 'd':
            cam_pos_world[1] -= pos_stride
        elif key == 'q':
            cam_pos_world[2] += pos_stride
        elif key == 'z':
            cam_pos_world[2] -= pos_stride
        elif key == '1':
            pcd_rot_vec[0] += rot_stride
        elif key == '2':
            pcd_rot_vec[0] -= rot_stride
        elif key == '3':
            pcd_rot_vec[1] += rot_stride
        elif key == '4':
            pcd_rot_vec[1] -= rot_stride
        elif key == '5':
            pcd_rot_vec[2] += rot_stride
        elif key == '6':
            pcd_rot_vec[2] -= rot_stride
        elif key == 'c':
            break

        # import pdb; pdb.set_trace()
        pcd_ori_world = qmult(qmult(qmult(axangle2quat([1, 0, 0], pcd_rot_vec[0]), 
            axangle2quat([0, 1, 0], pcd_rot_vec[1])), 
            axangle2quat([0, 0, 1], pcd_rot_vec[2])), 
            [1.0, 0.0, 0.0, 0.0])

        cam_ori_world = qmult(pcd_ori_world, cam_ori_world_init)
        cam_ori_world = [float(x) for x in cam_ori_world]
        print(f"Pos: {cam_pos_world}\nOri: {cam_ori_world}")
        br.sendTransform(tuple(cam_pos_world),
            tuple([cam_ori_world[1], cam_ori_world[2], cam_ori_world[3], cam_ori_world[0]]), 
            rospy.Time.now(), f"cam{cam_idx}_link", fixed_frame)
        
        rate.sleep()

    print(f"Final pos: {cam_pos_world}\nFinal ori: {cam_ori_world}")
    camera_pose_dict[f"cam_{cam_idx}"]["position"] = cam_pos_world
    camera_pose_dict[f"cam_{cam_idx}"]["orientation"] = cam_ori_world
    with open(os.path.join(data_dir, 'camera_pose_world.yml'), 'w') as f:
        yaml.dump(camera_pose_dict, f)


if __name__ == '__main__':
    main()
