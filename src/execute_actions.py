import glob
import numpy as np
import os
import readchar
import rospy
import sys

from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8, String
from timeit import default_timer as timer
from transforms3d.quaternions import *

import manipulate

robot = manipulate.ManipulatorSystem()

task_tool_mapping = {
    'cutting': 'planar_cutter', 
    'gripping': 'gripper', 
    'rolling': 'roller', 
    'pressing': 'stamp',
}

init_pose_seq = None
def init_pose_seq_callback(msg):
    global init_pose_seq
    init_pose_seq = msg.data.reshape(-1, 11, 14)

act_seq = None
def act_seq_callback(msg):
    global act_seq
    act_seq = msg.data.reshape(-1, 30, 12)

iter = 0
def iter_callback(msg):
    global iter
    iter = msg.data

command_str = ''
def command_callback(msg):
    global command_str
    command_str = msg.data


def main():
    global robot
    global init_pose_seq
    global act_seq
    global iter
    global task_type
    global command_str

    if len(sys.argv) < 2:
        print("Please enter the mode!")
        return

    rospy.init_node('execute_actions', anonymous=True)

    rospy.Subscriber("/init_pose_seq", numpy_msg(Floats), init_pose_seq_callback)
    rospy.Subscriber("/act_seq", numpy_msg(Floats), act_seq_callback)
    rospy.Subscriber("/iter", UInt8, iter_callback)
    rospy.Subscriber("/command", String, command_callback)

    mode = sys.argv[1]
    if mode == 'react':
        react()
    elif mode == 'replay':
        result_dir = readchar.readkey()
        if os.path.exists(result_dir):
            best_init_pose_seq = np.load(f"{result_dir}/best_init_pose_seq.npy", allow_pickle=True)
            best_act_seq = np.load(f"{result_dir}/best_act_seq.npy", allow_pickle=True)
            for task_name in task_tool_mapping.keys():
                if task_name in result_dir:
                    execute(task_name, best_init_pose_seq, best_act_seq)
                    break
        else:
            print("Result directory doesn't exist!")
    else:
        raise NotImplementedError


def execute(task_name, init_pose_seq, act_seq):
    if 'gripping' in task_name:
        mid_point = (init_pose_seq.shape[1] - 1) // 2
        grip_h = 0.175
        pregrip_dh = 0.1
        for i in range(init_pose_seq.shape[0]):
            grip_pos = (init_pose_seq[i, mid_point, :3] + init_pose_seq[i, mid_point, 3:]) / 2

            rot_z = np.arctan2(init_pose_seq[i, mid_point, 8] - init_pose_seq[i, mid_point, 1], 
                init_pose_seq[i, mid_point, 7] - init_pose_seq[i, mid_point, 0]) + np.pi / 4

            if rot_z > np.pi / 2:
                rot_z -= np.pi
            elif rot_z < -np.pi / 2:
                rot_z += np.pi

            end_pos_1 = init_pose_seq[i, mid_point, :3] + np.sum(act_seq[i, :, :3], axis=0)
            end_pos_2 = init_pose_seq[i, mid_point, 3:] + np.sum(act_seq[i, :, 6:9], axis=0)

            if 'asym' in task_name:
                grip_width = np.linalg.norm(end_pos_1 - end_pos_2) - 0.035521
            else:
                grip_width = np.linalg.norm(end_pos_1 - end_pos_2) - 0.038632

            robot.grip((grip_pos[0], grip_pos[1], rot_z), grip_h, pregrip_dh, grip_width)

    elif 'rolling' in task_name:
        preroll_dh = 0.07
        for i in range(init_pose_seq.shape[0]):
                rot_noise = np.random.uniform(-np.pi, np.pi)
    if rot_noise > np.pi / 2:
        rot_noise -= np.pi
    elif rot_noise < -np.pi / 2:
        rot_noise += np.pi

    roll_rot = [0.0, 0.0, rot_noise]
    roll_dist = 0.03 + roll_dist_noise_scale * np.random.rand()
    roll_delta = axangle2mat([0, 0, 1], roll_rot[2] + np.pi / 4) @ np.array([roll_dist, 0, 0]).T
            robot.roll(start_pos, roll_rot, end_pos, preroll_dh)

    elif 'pressing' in task_name:
        robot.press(press_pos, press_rot, prepress_dh)
    else:
        raise NotImplementedError


def react():
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        if len(command) != 0:
            command_list = command_str.split('_')
            command = command_list[0]
            if command == 'start':
                # take away the tool
                tool = task_tool_mapping[command_list[1]]
                if robot.tool_status[tool] == 'ready':
                    print(f"========== taking away {tool} ==========")
                    robot.take_away_tool(tool)
                    robot.tool_status[tool] = 'using'
                else:
                    print(f"========== ERROR: {tool} is being used! ==========")
            elif command == 'do':
                task_name = command_list[1]
                if init_pose_seq is not None and act_seq is not None:
                    execute(task_name, init_pose_seq, act_seq)
                else:
                    print("========== ERROR: Please publish init_pose_seq and act_seq ==========")
            elif command == 'switch':
                done = False
                for tool, status in robot.tool_status.items():
                    if status == 'using':
                        print(f"========== putting back {tool} ==========")
                        robot.put_back_tool(tool)
                        robot.tool_status[tool] = 'ready'
                        done = True
                        break
                
                if not done:
                    print("========== ERROR: No tool is being used! ==========")
                    continue
                
                tool = task_tool_mapping[command_list[1]]
                if robot.tool_status[tool] == 'ready':
                    print(f"========== taking away {tool} ==========")
                    robot.take_away_tool(tool)
                    robot.tool_status[tool] = 'using'
                else:
                    print(f"========== ERROR: {tool} is being used! ==========")
            elif command == 'end':
                done = False
                for tool, status in robot.tool_status.items():
                    if status == 'using':
                        print(f"========== putting back {tool} ==========")
                        robot.put_back_tool(tool)
                        robot.tool_status[tool] = 'ready'
                        done = True
                        break
                
                if not done:
                    print("========== ERROR: No tool is being used! ==========")
            else:
                print('========== ERROR: Unrecoganized command! ==========')

        rate.sleep()


if __name__ == "__main__":
    main()
