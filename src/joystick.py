# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy as np
import rospy
import torch

from timeit import default_timer as timer
from torchcontrol.transform import Rotation as R
from transforms3d.quaternions import *

from manipulate import ManipulatorSystem
from sensor_msgs.msg import Joy

robot = ManipulatorSystem()

# x, y, z
robot_bbox = [
    np.array([0.1, -0.3, 0.05]), 
    np.array([0.7, 0.3, 0.65])
]

tool_status = {
    'gripper': 'ready', 
    'roller': 'ready', 
    'planar_cutter': 'ready', 
    'circular_cutter': 'ready',
    'shovel': 'ready',
}

rest_pos = copy.deepcopy(robot.rest_pos)
pos_curr = [*rest_pos]
rot_curr = [0.0, 0.0, np.pi / 4]

init_pos_stride = 0.02
init_rot_stride = np.pi / 16
init_width_stride = 0.01
manipulating = False

# NEED TO initialize LT and RT before controllng with joystick
def joy_callback(msg):
    global pos_curr
    global rot_curr
    global manipulating

    # LB, RB, button 1, button 2, LT & LB
    if msg.buttons[4] or msg.buttons[5]:
        if msg.buttons[4]:
            if msg.axes[2] == 1.0 and msg.axes[5] == 1.0:
                tool = 'gripper'
            elif msg.axes[2] == -1.0 and msg.axes[5] == 1.0:
                tool = 'roller'
            elif msg.axes[2] == 1.0 and msg.axes[5] == -1.0:
                tool = 'planar_cutter'
            elif msg.axes[2] == -1.0 and msg.axes[5] == -1.0:
                tool = 'circular_cutter'
            else:
                raise NotImplementedError

        if msg.buttons[5]:
            if msg.axes[2] == 1.0 and msg.axes[5] == 1.0:
                tool = 'shovel'
            else:
                raise NotImplementedError

        if tool_status[tool] == 'ready':
            for k, v in tool_status.items():
                if k != tool and v == 'using':
                    print(f"Could not take {tool} while using {k}!")
                    return
            
            print(f"========== taking away {tool} ==========")
            take_away(tool)
            tool_status[tool] = 'using'
        else:
            print(f"========== putting back {tool} ==========")
            put_back(tool)
            tool_status[tool] = 'ready'

        # reset the pose
        pos_curr = [*rest_pos]
        rot_curr = [0.0, 0.0, np.pi / 4]

    elif 1 in msg.buttons[:4] or msg.axes[6] or msg.axes[7]:
        # pos_goal, quat_goal = robot.grasp_pose_to_pos_quat(pose_params, pose_h)
        pos_actual, quat_actual = robot.arm.get_ee_pose()
        # pos_error = np.linalg.norm(pos_actual - pos_goal)
        # quat_error = np.linalg.norm(quat_actual - quat_goal)
        # print(f"pos error: {pos_error}; quat_error: {quat_error}")
        # if pos_error > 0.05 or quat_error > 0.5:
        #     print(f'Waiting...')
        #     return

        lb_ratio = np.min((pos_actual.numpy() - robot_bbox[0]) / (robot.rest_pos.numpy() - robot_bbox[0]))
        ub_ratio = np.min((pos_actual.numpy() - robot_bbox[1]) / (robot.rest_pos.numpy() - robot_bbox[1]))
        pos_stride = min(lb_ratio, ub_ratio) * init_pos_stride
        # print(f"lower: {lb_ratio}; upper: {ub_ratio}; stride: {pos_stride}")

        if manipulating:
            time_to_go = 3.0
        else:
            time_to_go = 0.5

        # B
        if msg.buttons[1]:
            print(f"========== moving +x ==========")
            pos_curr[0] += pos_stride
        # X
        if msg.buttons[2]:
            print(f"========== moving -x ==========")
            pos_curr[0] -= pos_stride
        # Y
        if msg.buttons[3]:
            print(f"========== moving +y ==========")
            pos_curr[1] += pos_stride
        # A
        if msg.buttons[0]:
            print(f"========== moving -y ==========")
            pos_curr[1] -= pos_stride
        # up / down
        if msg.axes[7]:
            print(f"========== moving on {'+' if msg.axes[7] > 0 else '-'}z ==========")
            pos_curr[2] += msg.axes[7] / abs(msg.axes[7]) * pos_stride
        
        # left / right
        if msg.axes[6]:
            if msg.axes[2] == 1.0 and msg.axes[5] == 1.0:
                print(f"========== rotating on z ==========")
                rot_axis = 2 # z
                rot_stride = init_rot_stride
            elif msg.axes[2] == -1.0 and msg.axes[5] == 1.0:
                print(f"========== rotating on x ==========")
                rot_axis = 0 # x
                rot_stride = init_rot_stride
            elif msg.axes[2] == 1.0 and msg.axes[5] == -1.0:
                print(f"========== rotating on y ==========") 
                rot_axis = 1 # y
                rot_stride = init_rot_stride / 4

            rot_curr[rot_axis] += msg.axes[6] / abs(msg.axes[6]) * rot_stride
            if rot_curr[rot_axis] > np.pi / 2:
                rot_curr[rot_axis] -= np.pi
                time_to_go = 3.0
            elif rot_curr[rot_axis] < -np.pi / 2:
                rot_curr[rot_axis] += np.pi
                time_to_go = 3.0

        pos_curr = np.clip(pos_curr, robot_bbox[0], robot_bbox[1])
        print(f'pos_curr: {pos_curr}; rot_curr: {rot_curr}')

        pos_goal = torch.Tensor(pos_curr)
        rot_goal = (
            R.from_rotvec(torch.Tensor(rot_curr)) * R.from_quat(torch.Tensor([1, 0, 0, 0]))
        ).as_quat()
        robot.move_to(pos_goal, rot_goal, time_to_go=time_to_go)

    elif msg.buttons[6] or msg.buttons[7]:
        width_cur = robot.gripper.get_state().width
        # button 1
        if msg.buttons[6]:
            robot.open_gripper()
        # button 3
        if msg.buttons[7]:
            # for k, v in tool_status.items():
            #     if v == 'using':
            #         if k == 'gripper':
            #             width = 0.007
            #         else:
            #             width = 0.015
            #         break

            # customized grasp_speed and grasp_force
            width = width_cur - init_width_stride
            if width >= 0.0:
                robot.close_gripper(width, blocking=False, grasp_params=(0.01, 50.0))

    elif msg.axes[0]:
        print("========== start manipulating... ==========")
        manipulating = True

    elif msg.axes[3]:
        print("========== end manipulating... ==========")
        manipulating = False


def take_away(tool):
    if tool == 'gripper':
        robot.take_away(grasp_params=(0.415, 0.27, np.pi / 4), grasp_h=0.315, pregrasp_dh=0.01, grasp_width=0.01, lift_dh=0.1, loc='left')
    elif tool == 'roller':
        robot.take_away(grasp_params=(0.62, 0.19, -np.pi / 4), grasp_h=0.325, pregrasp_dh=0.01, grasp_width=0.015)
    elif tool == 'planar_cutter':
        robot.take_away(grasp_params=(0.62, 0.065, -np.pi / 4), grasp_h=0.325, pregrasp_dh=0.01, grasp_width=0.015)
    elif tool == 'circular_cutter':
        robot.take_away(grasp_params=(0.62, -0.1, -np.pi / 4), grasp_h=0.325, pregrasp_dh=0.01, grasp_width=0.015)
    elif tool == 'shovel':
        robot.take_away(grasp_params=(0.62, -0.225, -np.pi / 4), grasp_h=0.325, pregrasp_dh=0.01, grasp_width=0.015)
    else:
        raise NotImplementedError


def put_back(tool):
    if tool == 'gripper':
        robot.put_back_gripper(grasp_params=(0.415, 0.26, np.pi / 4), grasp_h=0.315, pregrasp_dh=0.05)
    elif tool == 'roller':
        robot.put_back(grasp_params=(0.615, 0.19, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.015)
    elif tool == 'planar_cutter':
        robot.put_back(grasp_params=(0.615, 0.065, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.015)
    elif tool == 'circular_cutter':
        robot.put_back(grasp_params=(0.615, -0.1, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.015)
    elif tool == 'shovel':
        robot.put_back(grasp_params=(0.615, -0.225, -np.pi / 4), grasp_h=0.32, pregrasp_dh=0.015)
    else:
        raise NotImplementedError


# Initialize interfaces
def main():
    rospy.init_node('grasp', anonymous=True)

    # subscribed to joystick inputs on topic "joy"
    rospy.Subscriber("joy", Joy, joy_callback)
    
    robot.reset()
    print("========== I'm ready! ==========")

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == "__main__":
    # Need to run `roscore`` and `rosrun joy joy_node`
    main()