
import numpy as np
import os
import readchar
import rospy
import open3d as o3d
import ros_numpy
import sys
import yaml

from act_from_param import *
from manipulate import ManipulatorSystem
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import *
from utils import get_center


robot = ManipulatorSystem()

with open('/scr/hxu/projects/RoboCook/config/tool_plan_params.yml', 'r') as f:
    tool_params = yaml.load(f, Loader=yaml.FullLoader)


pcd_signal = 0
center = None
def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global pcd_signal
    global center

    if pcd_signal == 1:
        pcd_msgs = [cam1_msg, cam2_msg, cam3_msg, cam4_msg]
        center = get_center(pcd_msgs)
        pcd_signal = 0


def random_explore(tool_name):
    global pcd_signal

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        print(f"start (s), run (r), end (e), or stop (c)?")
        key = readchar.readkey()
        print(key)
        if key == 's':
            for key, value in robot.tool_status.items():
                if value['status'] == 'using':
                    print(f"===== putting back {key} =====")
                    robot.put_back_tool(key)
                    break

            print(f"===== taking away {tool_name} =====")
            if robot.tool_status[tool_name]['status'] == 'ready':
                robot.take_away_tool(tool_name)
        elif key == 'r':
            pcd_signal = 1
            
            rate = rospy.Rate(100)
            while not rospy.is_shutdown():
                if pcd_signal == 0:
                    break

                rate.sleep()

            print(f"The center of the play_doh is {center}")

            if 'gripper' in tool_name:
                if 'plane' in tool_name:
                    random_grip(tool_name, 5, grip_width_min=0.02)
                else:
                    random_grip(tool_name, 5)
            elif 'press' in tool_name or 'punch' in tool_name:
                random_press(tool_name, 3)
            elif 'roller' in tool_name:
                random_roll(tool_name, 3)
            elif 'cutter_planar' in tool_name:
                random_cut_planar(tool_name)
            elif 'cutter_circular' in tool_name:
                print(f"===== Circular Cut: {center} =====")
                cut_circular(robot, center[:2])
            elif 'pusher' in tool_name:
                print(f"===== Push: {center} =====")
                push(robot, center[:2])
            elif 'spatula' in tool_name:
                params = [*center, 0.41, -0.29]
                print(f"===== Pick and Place: {params} =====")
                grip_width = 0.01 if 'large' in tool_name else 0.02
                pick_and_place(robot, params, grip_width)
            elif 'hook' in tool_name:
                print(f"===== Hook =====")
                hook(robot)
            else:
                raise NotImplementedError
        elif key == 'e':
            if robot.tool_status[tool_name]['status'] == 'using':
                print(f"===== putting back {tool_name} =====")
                robot.put_back_tool(tool_name)
        elif key == 'c':
            break
        else:
            print('Unrecoganized command!')

        rate.sleep()


def random_grip(tool_name, n_grips, grip_width_max=0.03, grip_width_min=0.01):
    for i in range(n_grips):
        dist_to_center = np.random.uniform(*tool_params[tool_name]['dist_range'])
        rot = np.random.uniform(*tool_params[tool_name]['rot_range'])
        grip_width = np.random.rand() * (grip_width_max - grip_width_min) + grip_width_min

        params = [*center, dist_to_center, rot, grip_width]
        print(f"===== Grip {i+1}: {params} =====")
        grip(robot, params)


def random_press(tool_name, n_presses):
    for i in range(n_presses):
        x_noise = tool_params[tool_name]['x_noise'] * (np.random.rand() * 2 - 1)
        y_noise = tool_params[tool_name]['y_noise'] * (np.random.rand() * 2 - 1)
        z_noise = tool_params[tool_name]['z_noise'] * (np.random.rand() * 2 - 1)
        press_pos = [center[0] + x_noise, center[1] + y_noise, 
            0.069 + 0.1034 + center[2] + z_noise]

        if 'circle' in tool_name:
            rot = 0
        else:
            rot = np.random.uniform(*tool_params[tool_name]['rot_range'])

        params = [*press_pos, rot]
        print(f"===== Press {i+1}: {params} =====")
        press(robot, params)


def random_roll(tool_name, n_rolls, roll_dist_noise=0.02):
    for i in range(n_rolls):
        x_noise = tool_params[tool_name]['x_noise'] * (np.random.rand() * 2 - 1)
        y_noise = tool_params[tool_name]['y_noise'] * (np.random.rand() * 2 - 1)
        z_noise = tool_params[tool_name]['z_noise'] * (np.random.rand() * 2 - 1)
        press_pos = [center[0] + x_noise, center[1] + y_noise, 
            0.089 + 0.1034 + center[2] + z_noise]

        rot = np.random.uniform(*tool_params[tool_name]['rot_range'])

        roll_dist = 0.04 + roll_dist_noise * np.random.rand()
        if np.random.randn() > 0.5:
            roll_dist = -roll_dist

        params = [*press_pos, rot, roll_dist]
        print(f"===== Roll {i+1}: {params} =====")
        roll(robot, params)


def random_cut_planar(pos_noise=0.01, rot_noise=np.pi/4):
    cut_x = center[0] + pos_noise * (np.random.rand() * 2 - 1)
    cut_y = center[1] + pos_noise * (np.random.rand() * 2 - 1)
    cut_rot = np.pi / 4 + rot_noise * np.random.rand()
    params = [cut_x, cut_y, cut_rot]
    print(f"===== Planar Cut: {params} =====")
    cut_planar(robot, params)


def main():
    if len(sys.argv) < 2:
        print("Please enter the tool name!")
        return

    tool_name = sys.argv[1]

    rospy.init_node('random_explore', anonymous=True)

    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/depth/color/points", PointCloud2), 
        Subscriber("/cam2/depth/color/points", PointCloud2), 
        Subscriber("/cam3/depth/color/points", PointCloud2), 
        Subscriber("/cam4/depth/color/points", PointCloud2)),
        queue_size=100,
        slop=0.2
    )

    tss.registerCallback(cloud_callback)

    random_explore(tool_name)


if __name__ == "__main__":
    main()
