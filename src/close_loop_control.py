import numpy as np
import os
import rosbag
import rospy
import sys
import timeit
import tf2_ros
import yaml
import cv2

from act_from_param import *
from cv_bridge import CvBridge
from datetime import datetime
from geometry_msgs.msg import TransformStamped
from manipulate import ManipulatorSystem
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import UInt8, Float32, String
from timeit import default_timer as timer
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import *
from utils import get_cube_center


robot = ManipulatorSystem()

data_path_signal = 0
data_path = ''
def path_callback(msg):
    global data_path
    global data_path_signal
    if data_path != msg.data:
        data_path = msg.data
        data_path_signal = 1
        print(f"[INFO] ROS data path: {data_path}")


param_seq = None
def param_seq_callback(msg):
    global param_seq
    # first number is n_actions
    param_seq = msg.data


command_str_signal = 0
command_str = ''
def command_callback(msg):
    global command_str
    global command_str_signal
    if command_str != msg.data:
        command_str = msg.data
        command_str_signal = 1
        print(f"[INFO] Command: {command_str}")


pcd_signal = 0
def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global pcd_signal
    global center

    if pcd_signal == 1:
        bag = rosbag.Bag(data_path + '.bag', 'w')
        bag.write('/cam1/depth/color/points', cam1_msg)
        bag.write('/cam2/depth/color/points', cam2_msg)
        bag.write('/cam3/depth/color/points', cam3_msg)
        bag.write('/cam4/depth/color/points', cam4_msg)

        # bag.write('/robot_pose', robot_pose_msg)
        # ee_pose = robot.get_ee_pose()
        # bag.write('/ee_pose', ee_pose)

        # gripper_width = robot.gripper.get_state().width
        # bag.write('/gripper_width', Float32(gripper_width))

        bag.close()

        print(f"[INFO] pcd recorded!")

        # cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        # debug_bag_path = os.path.join(cd, '..', 'raw_data', 'debug', 'state_0.bag')
        # os.system(f'cp {debug_bag_path} {data_path}")}')
        # print(f"Copied pcd at {os.path.basename(data_path)}...")

        pcd_signal = 0

    if pcd_signal == 2:
        get_cube(cam1_msg, cam2_msg, cam3_msg, cam4_msg)

        pcd_signal = 0


def run(param_seq):
    param_seq = param_seq.reshape(-1, 5)
    for i in range(len(param_seq)):
        param_seq_updated = [*param_seq[i]]

        print(f'===== Grip {i+1}: {param_seq_updated} =====')
        grip(robot, param_seq_updated)


def react():
    global param_seq
    global command_str
    global command_str_signal
    global data_path_signal
    global pcd_signal

    command_fb_pub = rospy.Publisher('/command_feedback', UInt8, queue_size=10)
    
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        if command_str_signal == 0: 
            continue

        command_list = command_str.split('.')
        command_time = command_list[0]
        command = command_list[1]

        if command == 'run':
            while param_seq is None:
                continue

            pcd_signal = 2
            run(param_seq)
            
            param_seq = None
            command_fb_pub.publish(UInt8(1))

        elif command == 'shoot':
            if len(command_list) > 2: 
                data_format_str = command_list[2]
            else:
                raise ValueError

            while data_path_signal == 0:
                continue

            data_path_signal = 0

            if 'pcd' in data_format_str:
                pcd_signal = 1

        else:
            print('========== ERROR: Unrecoganized command! ==========')

        command_str_signal = 0
        rate.sleep()


def main():
    global pcd_signal
    global data_path

    mode = 'react' if len(sys.argv) < 2 else sys.argv[1]
    
    rospy.init_node('close_loop_control', anonymous=True)

    rospy.Subscriber("/command", String, command_callback)
    rospy.Subscriber("/param_seq", numpy_msg(Floats), param_seq_callback)
    rospy.Subscriber("/raw_data_path", String, path_callback)
    
    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/depth/color/points", PointCloud2), 
        Subscriber("/cam2/depth/color/points", PointCloud2), 
        Subscriber("/cam3/depth/color/points", PointCloud2), 
        Subscriber("/cam4/depth/color/points", PointCloud2)),
        queue_size=100,
        slop=0.5
    )

    tss.registerCallback(cloud_callback)

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
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

    if mode == 'react':
        print("Reacting...")
        react()
    elif mode == 'replay':
        print("Replaying...")
        result_dir = input("Please enter the path of the solution: \n")
        if os.path.exists(result_dir):
            # param_seq = np.load(result_dir, allow_pickle=True)
            with open(result_dir, 'r') as f:
                param_seq_dict = yaml.load(f, Loader=yaml.FullLoader)
            
            for tool_name, param_seq in param_seq_dict.items():
                run(tool_name, np.array(param_seq))
        else:
            print("Result directory doesn't exist!")
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
