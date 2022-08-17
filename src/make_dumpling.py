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

pcd_signal = 0
center = None
def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global pcd_signal
    global center

    if pcd_signal == 1:
        pcd_msgs = [cam1_msg, cam2_msg, cam3_msg, cam4_msg]
        center = get_center(pcd_msgs)
        pcd_signal = 0


def wait_for_center():
    global pcd_signal

    pcd_signal = 1
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        if pcd_signal == 0:
            break

        rate.sleep()

    print(f"The center of the play_doh is {center}")


def make_dumpling(tool_name):
    global pcd_signal

    # step 1: planar cut to the right volume
    robot.take_away_tool('planar_cutter')
    wait_for_center()
    params = [] 
    cut_planar(robot, params)
    robot.put_back_tool('gripper_sym_plane')

    # step 2: grip the object to a cube
    robot.take_away_tool('gripper_sym_plane')
    wait_for_center()
    param_seq = []
    for params in param_seq:
        grip(robot, params)
    robot.put_back_tool('gripper_sym_plane')

    # step 3: press the object to a flat cube
    robot.take_away_tool('press_square')
    wait_for_center()
    param_seq = []
    for params in param_seq:
        press(robot, params)
    robot.put_back_tool('press_square')

    # step 4: roll the object to a flat surface
    robot.take_away_tool('roller_large')
    wait_for_center()
    param_seq = []
    for params in param_seq:
        roll(robot, params)
    robot.put_back_tool('roller_large')

    # step 5: circular cut
    robot.take_away_tool('cutter_circular')
    wait_for_center()
    cut_circular(robot, center[:2])
    robot.put_back_tool('cutter_circular')

    # step 6: push
    push(robot, center[:2])

    print(f"===== Pick and Place: {params} =====")
                grip_width = 0.01 if 'large' in tool_name else 0.02
                pick_and_place(robot, params, grip_width)
            elif 'hook' in tool_name:
                print(f"===== Hook =====")
                hook(robot)

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

    make_dumpling(tool_name)


if __name__ == "__main__":
    main()
