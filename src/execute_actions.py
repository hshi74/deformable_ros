import numpy as np
import os
import readchar
import rospy
import sys
import yaml

from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import UInt8, String
from timeit import default_timer as timer

from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import *

import manipulate

robot = manipulate.ManipulatorSystem()

task_tool_mapping = {
    'gripping_asym': 'gripper_asym', 
    'gripping_sym_rod': 'gripper_sym_rod',
    'gripping_sym_plane': 'gripper_sym_plane', 
    'rolling': 'roller', 
    'pressing_large': 'stamp_large',
    'pressing_small': 'stamp_small',
}

param_seq = None
def param_seq_callback(msg):
    global param_seq
    # first number is n_actions
    param_seq = msg.data

command_str = ''
def command_callback(msg):
    global command_str
    command_str = msg.data


def main():
    if len(sys.argv) < 2:
        print("Please enter the mode!")
        return

    rospy.init_node('execute_actions', anonymous=True)

    rospy.Subscriber("/param_seq", numpy_msg(Floats), param_seq_callback)
    rospy.Subscriber("/command", String, command_callback)

    mode = sys.argv[1]
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
            
            for task_name, param_seq in param_seq_dict.items():
                for task in task_tool_mapping.keys():
                    if task in task_name:
                        task_name = task
                        break
                run(task_name, np.array(param_seq))
        else:
            print("Result directory doesn't exist!")
    else:
        raise NotImplementedError


def run(task_name, param_seq):
    if 'gripping' in task_name:
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
    elif 'pressing' in task_name:
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
    elif 'rolling' in task_name:
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

    command_fb_pub = rospy.Publisher('/command_feedback', UInt8, queue_size=10)
    
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        if len(command_str) == 0: continue

        command_list = command_str.split('.')
        command = command_list[0]
        if len(command_list) > 1: 
            task_name = command_list[1]

        if command == 'run':
            import pdb; pdb.set_trace()
            selected_tool = task_tool_mapping[task_name]
            if robot.tool_status[selected_tool] == 'ready':
                for tool, status in robot.tool_status.items():
                    if status == 'using':
                        print(f"========== Putting back {tool} ==========")
                        robot.put_back_tool(tool)
                        break
        
                print(f"========== Taking away {selected_tool} ==========")
                robot.take_away_tool(selected_tool)

            if param_seq is not None:
                command_fb_pub.publish(UInt8(1))
                run(task_name, param_seq)
                param_seq = None

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


if __name__ == "__main__":
    main()
