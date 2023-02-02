import glob
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
from manipulate import ManipulatorSystem
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8, Float32, String
from timeit import default_timer as timer
from transforms3d.quaternions import *

robot = random_explore.robot

cd = os.path.dirname(os.path.realpath(sys.argv[0]))

hdd_str = '/media/hshi74/wd_drive/robocraft3d/raw_data'
data_path = os.path.join(hdd_str, f'gripper_sym_rod_robot_v3')

os.system('mkdir -p ' + f"{data_path}")

data_list = sorted(glob.glob(os.path.join(data_path, '*')))
if len(data_list) > 0:
    epi_prev = os.path.basename(data_list[-1]).split('_')[-1]
    episode = int(epi_prev[:-1].lstrip('0') + epi_prev[-1]) + 1
else:
    episode = 0


pcd_dy_signal = 0
def action_signal_callback(msg):
    global pcd_dy_signal
    if msg.data in [0, 1, 2]:
        pcd_dy_signal = msg.data


episode_signal = 0
seq = 0
def episode_signal_callback(msg):
    global episode_signal
    global episode
    global seq
    if episode_signal == 0 and msg.data == 1:
        episode += 1
        seq = 0

    episode_signal = msg.data


time_start, time_last, time_now, time_delta = 0.0, 0.0, 0.0, 0.1
n_actions = 0
action_counted = False
n_actions_max = 1
def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global pcd_dy_signal
    global time_start
    global time_last
    global time_now
    global n_actions
    global action_counted
    global seq

    def write_ros_bag(name):
        seq_path = os.path.join(data_path, f'ep_{str(episode).zfill(3)}', f'seq_{str(seq).zfill(3)}')
        os.system('mkdir -p ' + f"{seq_path}")
        bag = rosbag.Bag(os.path.join(seq_path, f'{name}.bag'), 'w')

        bag.write('/cam1/depth/color/points', cam1_msg)
        bag.write('/cam2/depth/color/points', cam2_msg)
        bag.write('/cam3/depth/color/points', cam3_msg)
        bag.write('/cam4/depth/color/points', cam4_msg)

        ee_pose = robot.get_ee_pose()
        bag.write('/ee_pose', ee_pose)

        gripper_width = robot.gripper.get_state().width
        bag.write('/gripper_width', Float32(gripper_width))

        bag.close()

    if pcd_dy_signal == 2:
        write_ros_bag(f'{-1:.3f}')
        print(f"Ep {episode} Seq {seq}: recorded pcd before rotation ...")

    elif pcd_dy_signal == 1:
        action_counted = False
        
        if time_start == 0.0:
            time_start = cam1_msg.header.stamp.to_sec()

        time_now = cam1_msg.header.stamp.to_sec() - time_start
        time_diff_1 = cam2_msg.header.stamp.to_sec() - time_start - time_now
        time_diff_2 = cam3_msg.header.stamp.to_sec() - time_start - time_now
        time_diff_3 = cam4_msg.header.stamp.to_sec() - time_start - time_now
        
        if time_now == 0.0 or time_now - time_last > time_delta:
            write_ros_bag(f'{time_now:.3f}')
            
            print(f"Ep {episode} Seq {seq}: recorded pcd at {round(time_now, 3)} " + \
                f"({round(time_diff_1, 3)}, {round(time_diff_2, 3)}, {round(time_diff_3, 3)})...")
            
            time_last = time_now

    elif pcd_dy_signal == 0:
        if time_start > 0 and not action_counted:
            n_actions += 1
            action_counted = True

        if n_actions >= n_actions_max:
            n_actions = 0
            seq += 1
            time_start = 0.0
            time_last = 0.0
            time_now = 0.0


def main():
    rospy.init_node("collect_dy_data", anonymous=True)

    rospy.Subscriber("/action_signal", UInt8, action_signal_callback)
    rospy.Subscriber("/episode_signal", UInt8, episode_signal_callback)
    
    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/depth/color/points", PointCloud2), 
        Subscriber("/cam2/depth/color/points", PointCloud2), 
        Subscriber("/cam3/depth/color/points", PointCloud2), 
        Subscriber("/cam4/depth/color/points", PointCloud2)),
        queue_size=100,
        slop=0.2
    )

    tss.registerCallback(cloud_callback)

    with open(os.path.join(cd, '..', 'env', 'camera_pose_world.yml'), 'r') as f:
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

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main()
