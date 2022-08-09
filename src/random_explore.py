import numpy as np
import random
import readchar
import rospy
import sys

from std_msgs.msg import UInt8
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import *

import manipulate

robot = manipulate.ManipulatorSystem()

def main():
    if len(sys.argv) < 2:
        print("Please enter the tool name!")
        return

    tool_name = sys.argv[1]

    rospy.init_node('random_explore', anonymous=True)

    collect_cls_data(tool_name)
    # collect_dy_data(tool_name)


def collect_cls_data(tool_name):
    cls_signal_pub = rospy.Publisher('/cls_signal', UInt8, queue_size=10)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        print(f"start (s), run (r), end (e), or stop (c)?")
        key = readchar.readkey()
        print(key)
        if 's' in key:
            for key, value in robot.tool_status.items():
                if value['status'] == 'using':
                    print(f"========== putting back {key} ==========")
                    robot.put_back_tool(key)
                    break

            print(f"========== taking away {tool_name} ==========")
            if robot.tool_status[tool_name]['status'] == 'ready':
                robot.take_away_tool(tool_name)
        elif 'r' in key:
            cls_signal_pub.publish(UInt8(1))
            if 'gripper' in tool_name:
                if 'plane' in tool_name:
                    random_grip(1, grip_width_min=0.01, grip_width_max=0.02, pregrip_dh=0.2)
                else:
                    random_grip(1, grip_width_min=0.005, grip_width_max=0.02, pregrip_dh=0.2)
            elif 'roller' in tool_name:
                random_roll(1, preroll_dh=0.2)
            elif 'presser' in tool_name:
                random_press(1, prepress_dh=0.2)
            elif 'cutter_planar' in tool_name:
                random_cut(1, cut_h=0.215, precut_dh=0.2, push_y=0.15)
            elif 'cutter_circular' in tool_name:
                random_cut(1, cut_h=0.17, pos_noise_scale=0, precut_dh=0.2, push_y=0)
            elif 'spatula' in tool_name:
                grip_width = 0.01 if 'large' in tool_name else 0.02
                robot.pick_and_place((0.4, -0.1, -np.pi / 4), 0.18, 0.2, 
                    (0.41, -0.29, -np.pi / 4), 0.22, 0.05, grip_width)
            elif 'hook' in tool_name:
                robot.hook_dumpling_clip([0.41, -0.2, 0.2])
            else:
                raise NotImplementedError
            
            cls_signal_pub.publish(UInt8(2))
        elif 'e' in key:
            if robot.tool_status[tool_name]['status'] == 'using':
                print(f"========== putting back {tool_name} ==========")
                robot.put_back_tool(tool_name)
        elif 'c' in key:
            break
        else:
            print('Unrecoganized command!')

        rate.sleep()


def collect_dy_data(tool_name):
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        print(f"start (s), run (r), end (e), or stop (c)?")
        key = readchar.readkey()
        print(key)
        if key == 's':
            # take away the tool
            if robot.tool_status[tool_name] == 'ready':
                print(f"========== taking away {tool_name} ==========")
                robot.take_away_tool(tool_name)
        elif key == 'r':
            if 'gripper' in tool_name:
                random_grip(5)
            elif 'roller' in tool_name:
                random_roll(2)
            elif 'presser' in tool_name:
                random_press(2)
            elif 'cutter_planar' in tool_name:
                random_cut(2, cut_h=0.215)
            elif 'cutter_circular' in tool_name:
                random_cut(2, cut_h=0.175)
            elif 'spatula' in tool_name:
                grip_width = 0.01 if 'large' in tool_name else 0.02
                robot.pick_and_place((0.4, -0.1, -np.pi / 4), 0.18, 0.1, 
                    (0.41, -0.29, -np.pi / 4), 0.22, 0.05, grip_width)
            elif 'hook' in tool_name:
                robot.hook_dumpling_clip([0.41, -0.2, 0.2])
            else:
                raise NotImplementedError
        elif key == 'e':
            if robot.tool_status[tool_name] == 'using':
                print(f"========== putting back {tool_name} ==========")
                robot.put_back_tool(tool_name)
        elif key == 'c':
            break
        else:
            print('Unrecoganized command!')

        rate.sleep()


def random_cut(n_cuts, cut_h, pos_noise_scale=0.03, precut_dh=0.07, push_y=0.03):
    for i in range(n_cuts):
        pos_noise_x = pos_noise_scale * (np.random.rand() * 2 - 1)
        pos_noise_y = pos_noise_scale * (np.random.rand() * 2 - 1)
        pos_noise_x = 0
        cut_pos = [0.4 + pos_noise_x, -0.1 + pos_noise_y, cut_h]
        cut_rot = [0.0, -0.05, np.pi / 4]
        robot.cut(cut_pos, cut_rot, precut_dh, push_y=push_y)


def random_grip(n_grips, pos_noise_scale=0.02, grip_width_max=0.02, grip_width_min=0.005, pregrip_dh=0.1):
    # Perform gripping
    grip_pos = np.array([0.4, -0.1])
    
    grip_h = 0.18
    for i in range(n_grips):
        # sample grip
        pos_noise = pos_noise_scale * (np.random.rand() * 2 - 1)
        rot_noise = np.random.uniform(-np.pi, np.pi)
        
        if rot_noise > np.pi / 2:
            rot_noise -= np.pi
        elif rot_noise < -np.pi / 2:
            rot_noise += np.pi

        grip_width = np.random.rand() * (grip_width_max - grip_width_min) + grip_width_min

        grip_pos_x = grip_pos[0] - pos_noise * np.sin(rot_noise - np.pi / 2)
        grip_pos_y = grip_pos[1] + pos_noise * np.cos(rot_noise - np.pi / 2)

        grip_params = (grip_pos_x, grip_pos_y, rot_noise)
        # Perform grip
        print(f"===== Grip {i+1}: {grip_params} and {grip_width} =====")
        robot.grip((grip_pos_x, grip_pos_y, rot_noise), grip_h, pregrip_dh, grip_width)


def random_roll(n_rolls, pos_noise_scale=0.01, roll_h_noise_scale=0.01, roll_dist_noise_scale=0.04, preroll_dh=0.07):
    for i in range(n_rolls):
        pos_noise_x = pos_noise_scale * (np.random.rand() * 2 - 1)
        pos_noise_y = pos_noise_scale * (np.random.rand() * 2 - 1)
        pos_noise_z = roll_h_noise_scale * (np.random.rand() * 2 - 1)
        start_pos = [0.4 + pos_noise_x, -0.1 + pos_noise_y, 0.205 + pos_noise_z]

        rot_noise = np.random.uniform(-np.pi, np.pi)
        if rot_noise > np.pi / 2:
            rot_noise -= np.pi
        elif rot_noise < -np.pi / 2:
            rot_noise += np.pi

        roll_rot = [0.0, 0.0, rot_noise]
        roll_dist = 0.03 + roll_dist_noise_scale * np.random.rand()
        roll_delta = axangle2mat([0, 0, 1], roll_rot[2] + np.pi / 4) @ np.array([roll_dist, 0, 0]).T
        # if np.dot(np.array([pos_noise_x, pos_noise_y, pos_noise_z]) * roll_delta) > 0:
        if np.random.randn() > 0.5:
            roll_delta = -roll_delta
        
        end_pos = start_pos + roll_delta

        print(f'Roll: \n\tstart_pos: {start_pos}\n\troll_rot: {rot_noise}\n\troll_dist:{roll_dist}\n\t preroll_dh: {preroll_dh}')
        robot.roll(start_pos, roll_rot, end_pos, preroll_dh)


def random_press(n_presses, pos_noise_scale=0.02, press_h_noise_scale=0.015, prepress_dh=0.1):
    for i in range(n_presses):
        pos_noise_x = pos_noise_scale * (np.random.rand() * 2 - 1)
        pos_noise_y = pos_noise_scale * (np.random.rand() * 2 - 1)
        pos_noise_z = press_h_noise_scale * (np.random.rand() * 2 - 1)
        press_pos = [0.4 + pos_noise_x, -0.1 + pos_noise_y, 0.19 + pos_noise_z]

        rot_noise = np.random.uniform(-np.pi, np.pi)
        if rot_noise > np.pi / 2:
            rot_noise -= np.pi
        elif rot_noise < -np.pi / 2:
            rot_noise += np.pi

        press_rot = [0.0, 0.0, rot_noise]
        print(f'Press: \n\tpress_pos: {press_pos}\n\tpress_rot: {rot_noise}\n\t\n\t prepress_dh: {prepress_dh}')
        robot.press(press_pos, press_rot, prepress_dh)


if __name__ == "__main__":
    main()
