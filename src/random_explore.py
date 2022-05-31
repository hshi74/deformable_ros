import numpy as np
import readchar
import rospy
import sys

from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import *

import manipulate

robot = manipulate.ManipulatorSystem()

task_tool_mapping = {
    'cutting': 'planar',
    'gripping_sym': 'gripper_sym', 
    'gripping_asym': 'gripper_asym', 
    'rolling': 'roller', 
    'pressing': 'stamp',
}

# 0 -> uninitialized / pause; 1 -> start; 2 -> stop
signal = 0

# Initialize interfaces
def main():
    global signal

    if len(sys.argv) < 2:
        print("Please enter the task name!")
        return

    task_name = sys.argv[1]
    tool = task_tool_mapping[task_name]
    rospy.init_node(task_name, anonymous=True)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        print(f"start (s), run (r), end (e), or stop (c)?")
        key = readchar.readkey()
        print(key)
        if key == 's':
            # take away the tool
            if robot.tool_status[tool] == 'ready':
                print(f"========== taking away {tool} ==========")
                robot.take_away_tool(tool)
        elif key == 'r':
            if task_name == 'cutting':
                random_cut()
            elif 'gripping' in task_name:
                random_grip(3)
            elif task_name == 'rolling':
                random_roll()
            elif task_name == 'pressing':
                random_press(3)
            else:
                raise NotImplementedError
        elif key == 'e':
            if robot.tool_status[tool] == 'using':
                print(f"========== putting back {tool} ==========")
                robot.put_back_tool(tool)
        elif key == 'c':
            break
        else:
            print('Unrecoganized command!')

        rate.sleep()


def random_cut(pos_noise_scale=0.03):
    pos_noise_y = pos_noise_scale * (np.random.rand() * 2 - 1)
    print(pos_noise_y)
    cut_pos = [0.4, -0.1 + pos_noise_y, 0.23]
    cut_rot = [0.0, -0.05, np.pi / 4]
    precut_dh = 0.07
    robot.cut(cut_pos, cut_rot, precut_dh)


def random_grip(n_grips, pos_noise_scale=0.01, grip_width_noise=0.01):
    # Perform gripping
    grip_pos = np.array([0.4, -0.1])
    
    grip_h = 0.175
    pregrip_dh = 0.1
    for i in range(n_grips):
        # sample grip
        pos_noise_x = pos_noise_scale * (np.random.rand() * 2 - 1)
        pos_noise_y = pos_noise_scale * (np.random.rand() * 2 - 1)

        rot_noise = np.random.uniform(-np.pi, np.pi)

        grip_width = np.random.rand() * grip_width_noise + 0.005
        
        if rot_noise > np.pi / 2:
            rot_noise -= np.pi
        elif rot_noise < -np.pi / 2:
            rot_noise += np.pi

        grip_params = (grip_pos[0] + pos_noise_x, grip_pos[1] + pos_noise_y, rot_noise)

        # Perform grip
        print(f"===== Grip {i+1}: {grip_params} and {grip_width} =====")
        robot.grip(grip_params, grip_h, pregrip_dh, grip_width)


def random_roll(pos_noise_scale=0.01, roll_h_noise_scale=0.01, roll_dist_noise_scale=0.04):
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
    preroll_dh = 0.07

    print(f'Roll: \n\tstart_pos: {start_pos}\n\troll_rot: {rot_noise}\n\troll_dist:{roll_dist}\n\t preroll_dh: {preroll_dh}')
    robot.roll(start_pos, roll_rot, end_pos, preroll_dh)


def random_press(n_presses, pos_noise_scale=0.02, press_h_noise_scale=0.015):
    prepress_dh = 0.1
    for i in range(n_presses):
        pos_noise_x = pos_noise_scale * (np.random.rand() * 2 - 1)
        pos_noise_y = pos_noise_scale * (np.random.rand() * 2 - 1)
        pos_noise_z = press_h_noise_scale * (np.random.rand() * 2 - 1)
        press_pos = [0.4 + pos_noise_x, -0.1 + pos_noise_y, 0.175 + pos_noise_z]

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
