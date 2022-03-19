# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import rospy

from timeit import default_timer as timer
from transforms3d.quaternions import *

from manipulator import ManipulatorSystem
from sensor_msgs.msg import Joy

robot = ManipulatorSystem()

tool_status = {
    'gripper': 'idle', 
    'roller': 'idle', 
    'planar_cutter': 'idle', 
    'circular_cutter': 'idle',
    'stamp': 'idle',
}

mid_point = np.array([0.4, 0.0])
pose_params = [mid_point[0], mid_point[1], np.pi / 4]
pose_h = 0.4
pos_stride = 0.02
pos_h_stride = 0.02
rot_stride = np.pi / 16


def joy_callback(msg):
    global pose_params
    global pose_h

    # LB, RB, LT, RT
    if msg.buttons[4] or msg.buttons[5] or msg.buttons[6] or msg.buttons[7]:
        if msg.buttons[4]:
            tool = 'roller'
        elif msg.buttons[5]:
            tool = 'planar_cutter'
        elif msg.buttons[6]:
            tool = 'circular_cutter'
        elif msg.buttons[7]:
            tool = 'stamp'

        if tool_status[tool] == 'idle':
            for k, v in tool_status.items():
                if k != tool and v == 'using':
                    print(f"Could not take another tool while using {tool}!")
                    return
            take_away(tool)
            tool_status[tool] = 'using'
        else:
            put_back(tool)
            tool_status[tool] = 'idle'

    elif 1 in msg.buttons[:4] or msg.axes[6] or msg.axes[7]:
        # pos_goal, quat_goal = robot.grasp_pose_to_pos_quat(pose_params, pose_h)
        # pos_actual, quat_actual = robot.arm.get_ee_pose()
        # pos_error = np.linalg.norm(pos_actual - pos_goal)
        # quat_error = np.linalg.norm(quat_actual - quat_goal)
        # print(f"pos error: {pos_error}; quat_error: {quat_error}")
        # if pos_error > 0.05 or quat_error > 0.5:
        #     print(f'Waiting...')
        #     return

        # B
        if msg.buttons[1]:
            pose_params[0] += pos_stride
            pose_params[0] = min(max(pose_params[0], 0.1), 0.7)
            time_to_go = 0.1
        
        # X
        if msg.buttons[2]:
            pose_params[0] -= pos_stride
            pose_params[0] = min(max(pose_params[0], 0.1), 0.7)
            time_to_go = 0.1
        
        # Y
        if msg.buttons[3]:
            pose_params[1] += pos_stride
            pose_params[1] = min(max(pose_params[1], -0.3), 0.3)
            time_to_go = 0.1
        
        # A
        if msg.buttons[0]:
            pose_params[1] -= pos_stride
            pose_params[1] = min(max(pose_params[1], -0.3), 0.3)
            time_to_go = 0.1
        
        if msg.axes[6]:
            pose_params[2] += msg.axes[6] / abs(msg.axes[6]) * rot_stride
            time_to_go = 0.2
        
        if msg.axes[7]:
            pose_h += msg.axes[7] / abs(msg.axes[7]) * pos_h_stride
            pose_h = min(max(pose_h, 0.05), 0.65)
            time_to_go = 0.5

        print(f'pose_params: {pose_params}; pose_h: {pose_h}')

        pose_goal = robot.grasp_pose_to_pos_quat(pose_params, pose_h)
        robot.set_policy(robot.gain_dict['high']['Kx'], robot.gain_dict['high']['Kxd'])
        robot.move_to(*pose_goal, time_to_go=time_to_go)


def take_away(tool):
    if tool == 'gripper':
        robot.take_away(grasp_params=(0.415, 0.27, np.pi / 4), grasp_h=0.31, pregrasp_dh=0.01, grasp_width=0.03, lift_dh=0.1, loc='left')
    elif tool == 'roller':
        robot.take_away(grasp_params=(0.615, 0.19, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.01, grasp_width=0.015)
    elif tool == 'planar_cutter':
        robot.take_away(grasp_params=(0.615, 0.065, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.01, grasp_width=0.015)
    elif tool == 'circular_cutter':
        robot.take_away(grasp_params=(0.615, -0.1, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.01, grasp_width=0.015)
    elif tool == 'stamp':
        robot.take_away(grasp_params=(0.615, -0.225, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.01, grasp_width=0.015)
    else:
        raise NotImplementedError


def put_back(tool):
    if tool == 'gripper':
        robot.put_back_gripper(grasp_params=(0.415, 0.26, np.pi / 4), grasp_h=0.31, pregrasp_dh=0.05)
    elif tool == 'roller':
        robot.put_back(grasp_params=(0.615, 0.19, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.015)
    elif tool == 'planar_cutter':
        robot.put_back(grasp_params=(0.615, 0.065, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.015)
    elif tool == 'circular_cutter':
        robot.put_back(grasp_params=(0.615, -0.1, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.015)
    elif tool == 'stamp':
        robot.put_back(grasp_params=(0.615, -0.225, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.015)
    else:
        raise NotImplementedError


# Initialize interfaces
def main():
    rospy.init_node('grasp', anonymous=True)

    # subscribed to joystick inputs on topic "joy"
    rospy.Subscriber("joy", Joy, joy_callback)
    
    robot.reset()
    print("Ready...")

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == "__main__":
    # Need to run `roscore`` and `rosrun joy joy_node`
    main()