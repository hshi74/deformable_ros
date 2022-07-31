import execute_actions
import glob
import manipulate
import numpy as np
import os
import random_explore
import rosbag
import rospy
import sys
import timeit
import tf2_ros
import yaml

from datetime import datetime
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8, Float32, String
from timeit import default_timer as timer
from transforms3d.quaternions import *


def signal_callback(msg):
    global signal
    signal = msg.data


data_path = ''
def path_callback(msg):
    global data_path
    global signal
    if data_path != msg.data:
        if '0' in os.path.basename(msg.data):
            signal = 1
        data_path = msg.data
        print(f"MPC state data path: {data_path}")


# signal 0 -> uninitialized / pause; 1 -> start; 2 -> stop
signal = 0
time_start, time_last, time_now, time_delta = 0.0, 0.0, 0.0, 0.1
n_actions = 0
action_counted = False
n_actions_max = 1
tiral = 0
mode = ''
robot = None
def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global signal
    global time_start
    global time_last
    global time_now
    global n_actions
    global action_counted
    global trial

    if mode == 'explore':
        if signal == 1:
            action_counted = False
            
            if time_start == 0.0:
                time_start = cam1_msg.header.stamp.to_sec()

            time_now = cam1_msg.header.stamp.to_sec() - time_start
            time_diff_1 = cam2_msg.header.stamp.to_sec() - time_start - time_now
            time_diff_2 = cam3_msg.header.stamp.to_sec() - time_start - time_now
            time_diff_3 = cam4_msg.header.stamp.to_sec() - time_start - time_now
            
            if time_now == 0.0 or time_now - time_last > time_delta:
                trial_path = os.path.join(data_path, str(trial).zfill(3))
                os.system('mkdir -p ' + f"{trial_path}")
                bag = rosbag.Bag(os.path.join(trial_path, f'{time_now:.3f}.bag'), 'w')

                bag.write('/cam1/depth/color/points', cam1_msg)
                bag.write('/cam2/depth/color/points', cam2_msg)
                bag.write('/cam3/depth/color/points', cam3_msg)
                bag.write('/cam4/depth/color/points', cam4_msg)

                ee_pose = robot.get_ee_pose()
                bag.write('/ee_pose', ee_pose)

                gripper_width = robot.gripper.get_state().width
                bag.write('/gripper_width', Float32(gripper_width))

                bag.close()

                print(f"Trial {trial}: recorded pcd at {round(time_now, 3)} " + \
                    f"({round(time_diff_1, 3)}, {round(time_diff_2, 3)}, {round(time_diff_3, 3)})...")
                
                time_last = time_now

        if signal == 0:
            if time_start > 0 and not action_counted:
                n_actions += 1
                action_counted = True

            if n_actions >= n_actions_max:
                n_actions = 0
                trial += 1
                time_start = 0.0
                time_last = 0.0
                time_now = 0.0

    elif mode == 'control':
        if signal == 1:
            bag = rosbag.Bag(data_path, 'w')
            bag.write('/cam1/depth/color/points', cam1_msg)
            bag.write('/cam2/depth/color/points', cam2_msg)
            bag.write('/cam3/depth/color/points', cam3_msg)
            bag.write('/cam4/depth/color/points', cam4_msg)

            # bag.write('/robot_pose', robot_pose_msg)
            ee_pose = robot.get_ee_pose()
            bag.write('/ee_pose', ee_pose)

            gripper_width = robot.gripper.get_state().width
            bag.write('/gripper_width', Float32(gripper_width))

            bag.close()

            print(f"Recorded pcd at {os.path.basename(data_path)}...")

            # cd = os.path.dirname(os.path.realpath(sys.argv[0]))
            # debug_bag_path = os.path.join(cd, '..', 'raw_data', 'debug', 'state_0.bag')
            # os.system(f'cp {debug_bag_path} {data_path}")}')
            # print(f"Copied pcd at {os.path.basename(data_path)}...")

            signal = 0

    else:
        raise NotImplementedError


def main():
    global signal
    global trial
    global mode
    global data_path
    global robot

    if len(sys.argv) < 2:
        print("Please enter the mode!")
        exit()

    rospy.init_node('vision_sensor', anonymous=True)

    rospy.Subscriber("/signal", UInt8, signal_callback)
    rospy.Subscriber("/raw_data_path", String, path_callback)
    
    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/depth/color/points", PointCloud2), 
        Subscriber("/cam2/depth/color/points", PointCloud2), 
        Subscriber("/cam3/depth/color/points", PointCloud2), 
        Subscriber("/cam4/depth/color/points", PointCloud2)),
        queue_size=100,
        slop=0.2
    )

    tss.registerCallback(cloud_callback)

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    mode = sys.argv[1] # explore, record, or control
    if mode == 'explore':
        robot = random_explore.robot
        # task_name = 'gripping_sym_plane_robot_v3'
        # task_name = 'gripping_asym_robot_v3'
        # task_name = 'rolling_small_robot_v1'
        task_name = 'pressing_small_robot_v1'
        data_path = os.path.join(cd, '..', 'raw_data', task_name)
        os.system('mkdir -p ' + f"{data_path}")
        data_list = sorted(glob.glob(os.path.join(data_path, '*')))
        if len(data_list) == 0:
            trial = 0
        else:
            trial = int(os.path.basename(data_list[-1]).lstrip('0')) + 1
    elif mode == 'record':
        robot = manipulate.ManipulatorSystem()
    elif mode == 'control':
        robot = execute_actions.robot
    else:
        raise NotImplementedError

    with open(os.path.join(os.path.join(cd, '..', 'env'), 'camera_pose_world.yml'), 'r') as f:
        cam_pose_dict = yaml.load(f, Loader=yaml.FullLoader)

    static_br = tf2_ros.StaticTransformBroadcaster()
    static_ts_list = []
    for i in range(1, len(cam_pose_dict) + 1):
        static_ts = TransformStamped()
        static_ts.header.frame_id = 'panda_link0'
        static_ts.child_frame_id = f"cam{i}_link"

        static_ts.transform.translation.x = cam_pose_dict[f"cam_{i}"]["position"][0]
        static_ts.transform.translation.y = cam_pose_dict[f"cam_{i}"]["position"][1]
        static_ts.transform.translation.z = cam_pose_dict[f"cam_{i}"]["position"][2]

        static_ts.transform.rotation.x = cam_pose_dict[f"cam_{i}"]["orientation"][1]
        static_ts.transform.rotation.y = cam_pose_dict[f"cam_{i}"]["orientation"][2]
        static_ts.transform.rotation.z = cam_pose_dict[f"cam_{i}"]["orientation"][3]
        static_ts.transform.rotation.w = cam_pose_dict[f"cam_{i}"]["orientation"][0]
        
        static_ts_list.append(static_ts)

    static_br.sendTransform(static_ts_list)

    print(f"Ready to {mode}!")
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        if signal == 2: break
        rate.sleep()


if __name__ == '__main__':
    main()
