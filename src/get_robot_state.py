# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import rospy
import sys
import yaml

from polymetis import RobotInterface, GripperInterface
from sensor_msgs.msg import JointState
from transforms3d.quaternions import *


def get_joint_states(robot, gripper):
    joint_pos = robot.get_joint_angles()
    joint_vel = robot.get_joint_velocities()
    gripper_state = gripper.get_state()
    return joint_pos.tolist(), joint_vel.tolist(), gripper_state.width


def get_robot_state(robot, write=False):
    # Reset
    # robot.go_home()
    # state_log = robot.set_joint_positions(joint_pos_desired, time_to_go=2.0)
    
    robot_state = {}
    # Get joint positions
    # joint_pos, joint_vel, gripper_state = get_joint_states(robot, gripper)
    # print(f"Current joint positions: {joint_pos}")
    # print(f"Current gripper state: {gripper_state}")
    # robot_state['joint_positions'] = joint_pos
    # robot_state['gripper_state'] = gripper_state
    
    # Get ee pose
    ee_pos, ee_quat = robot.pose_ee()
    ee_ori = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])
    print(f"Current ee position: {ee_pos}")
    print(f"Current ee orientation: {ee_ori}  (wxyz)")
    robot_state['ee_pos'] = ee_pos.tolist()
    robot_state['ee_ori'] = ee_ori.tolist()

    ee_fingertip_T_mat = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1124], [0, 0, 0, 1]])
    fingertip_pos = (quat2mat(ee_ori) @ ee_fingertip_T_mat[:3, 3].T).T + ee_pos.numpy()
    fingertip_ori = qmult(ee_ori, mat2quat(ee_fingertip_T_mat[:3, :3]))
    # print(quat2mat(fingertip_ori))
    print(f"Current fingertip position: {fingertip_pos}")
    print(f"Current fingertip orientation: {fingertip_ori}  (wxyz)")
    robot_state['fingertip_pos'] = fingertip_pos.tolist()
    robot_state['fingertip_ori'] = fingertip_ori.tolist()

    if write:
        cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        with open(os.path.join(cd, '..', 'env', 'robot_state.yml'), 'w') as f:
            yaml.dump(robot_state, f)


def main():
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="192.168.0.2",
        enforce_version=False,
    )

    # gripper = GripperInterface(
    #     ip_address="192.168.0.2",
    # )

    get_robot_state(robot, write=True)


if __name__ == "__main__":
    main()
