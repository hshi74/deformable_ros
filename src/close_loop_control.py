import manipulate
import numpy as np
import os
import rosbag
import rospy
import sys
import timeit
import tf2_ros
import yaml
import cv2

from cv_bridge import CvBridge
from datetime import datetime
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import UInt8, Float32, String
from timeit import default_timer as timer
from transforms3d.quaternions import *


robot = manipulate.ManipulatorSystem()

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

command_str = ''
def command_callback(msg):
    global command_str
    command_str = msg.data


pcd_signal = 0
def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global pcd_signal

    if pcd_signal == 1:
        bag = rosbag.Bag(os.path.join(data_path, 'pcd.bag'), 'w')
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

        print(f"[INFO] pcd recorded!")

        # cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        # debug_bag_path = os.path.join(cd, '..', 'raw_data', 'debug', 'state_0.bag')
        # os.system(f'cp {debug_bag_path} {data_path}")}')
        # print(f"Copied pcd at {os.path.basename(data_path)}...")

        pcd_signal = 0


img_signal = 0
def image_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global img_signal

    if img_signal == 1:
        image_msgs = [cam1_msg, cam2_msg, cam3_msg, cam4_msg]
        br = CvBridge()
        for i in range(len(image_msgs)):
            # Convert ROS Image message to OpenCV image
            img_bgr = br.imgmsg_to_cv2(image_msgs[i])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            cv2.imwrite(os.path.join(data_path, f'cam_{i+1}.png'), img_rgb)

        print(f"[INFO] images written!")

        img_signal = 0


def run(tool_name, param_seq):
    if 'gripper' in tool_name:
        param_seq = param_seq.reshape(-1, 3)
        for i in range(len(param_seq)):
            grip_h = 0.18
            pregrip_dh = 0.1
            pos_noise, rot_noise, grip_width = param_seq[i]
            grip_pos_x = 0.4 - pos_noise * np.sin(rot_noise - np.pi / 2)
            grip_pos_y = -0.1 + pos_noise * np.cos(rot_noise - np.pi / 2)
            print(f'===== Grip {i+1}: {param_seq[i]} =====')
            if i == len(param_seq) - 1:
                grip_mode = 'react'
            else:
                grip_mode = 'na'
            robot.grip((grip_pos_x, grip_pos_y, rot_noise), grip_h, pregrip_dh, grip_width, mode=grip_mode)
    elif 'press' in tool_name or 'punch' in tool_name:
        param_seq = param_seq.reshape(-1, 4)
        for i in range(len(param_seq)):
            prepress_dh = 0.1
            press_pos_x, press_pos_y, press_pos_z, rot_noise = param_seq[i]
            print(f'===== Press {i+1}: {param_seq[i]} =====')
            if i == len(param_seq) - 1:
                press_mode = 'react'
            else:
                press_mode = 'na'
            robot.press((press_pos_x, press_pos_y, press_pos_z), rot_noise, prepress_dh, mode=press_mode)
    elif 'roller' in tool_name:
        param_seq = param_seq.reshape(-1, 5)
        for i in range(len(param_seq)):
            preroll_dh = 0.07
            roll_pos_x, roll_pos_y, roll_pos_z, rot_noise, roll_dist = param_seq[i]
            start_pos = [roll_pos_x, roll_pos_y, roll_pos_z]
            roll_rot = [0.0, 0.0, rot_noise]
            roll_delta = axangle2mat([0, 0, 1], roll_rot[2] + np.pi / 4) @ np.array([roll_dist, 0, 0]).T
            end_pos = start_pos + roll_delta
            print(f'===== Roll {i+1}: {param_seq[i]} =====')
            if i == len(param_seq) - 1:
                roll_mode = 'react'
            else:
                roll_mode = 'na'
            robot.roll(start_pos, roll_rot, end_pos, preroll_dh, mode=roll_mode)
    else:
        raise NotImplementedError


def react():
    global param_seq
    global command_str
    global data_path_signal
    global pcd_signal
    global img_signal

    command_fb_pub = rospy.Publisher('/command_feedback', UInt8, queue_size=10)
    
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        if len(command_str) == 0: continue

        command_list = command_str.split('.')
        command = command_list[0]

        if command == 'run':
            if len(command_list) > 1: 
                tool_name = command_list[1]
            else:
                raise ValueError

            import pdb; pdb.set_trace()
            if robot.tool_status[tool_name] == 'ready':
                for tool, status in robot.tool_status.items():
                    if status == 'using':
                        print(f"========== Putting back {tool} ==========")
                        robot.put_back_tool(tool)
                        break
        
                print(f"========== Taking away {tool_name} ==========")
                robot.take_away_tool(tool_name)

            if param_seq is not None:
                command_fb_pub.publish(UInt8(1))
                run(tool_name, param_seq)
                param_seq = None

        elif command == 'shoot':
            if len(command_list) > 1: 
                data_format_str = command_list[1]
            else:
                raise ValueError

            while data_path_signal == 0:
                continue

            data_path_signal = 0

            if 'pcd' in data_format_str:
                pcd_signal = 1
            if 'rgb' in data_format_str:
                img_signal = 1

        elif command == 'end':
            for tool, status in robot.tool_status.items():
                if status == 'using':
                    print(f"========== Putting back {tool} ==========")
                    command_fb_pub.publish(UInt8(1))
                    robot.put_back_tool(tool)
                    break

        else:
            print('========== ERROR: Unrecoganized command! ==========')

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
        slop=0.2
    )

    tss.registerCallback(cloud_callback)

    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/color/image_raw", Image), 
        Subscriber("/cam2/color/image_raw", Image), 
        Subscriber("/cam3/color/image_raw", Image), 
        Subscriber("/cam4/color/image_raw", Image)),
        queue_size=10,
        slop=0.1
    )

    tss.registerCallback(image_callback)

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
