# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import rospy
import tempfile
import time
import torch
torch.set_printoptions(precision=4, profile='short', sci_mode=False)
import torchcontrol as toco

from geometry_msgs.msg import Pose
from polymetis import RobotInterface, GripperInterface
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from timeit import default_timer as timer
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from transforms3d.quaternions import *

# Grasp params
mid_point = np.array([0.437, 0.0])
rest_pose = ([mid_point[0], 0.0, 0.6], [1.0, 0.0, 0.0, 0.0])
grasp_speed = 0.02
grasp_force = 10.0

planner_dt = 0.01

ee_fingertip_T_mat = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
gripper_ft_trans = np.array([0.0, 0.019, 0.042192])


class ManipulatorSystem:
    def __init__(self):
        self.arm = RobotInterface(
            ip_address="192.168.0.2",
            enforce_version=False,
        )

        # self.update_urdf("/scr/hxu/catkin_ws/src/RoboCook_ROS/urdf/panda_arm_hand.urdf")

        self.gain_dict = {
            'low': {
                'Kx': torch.tensor([300., 300., 300., 40., 40., 40.]),
                'Kxd': torch.tensor([35., 35., 35., 14., 14., 14.])
            },
            'high': {
                'Kx': torch.tensor([600., 600., 1200., 160., 160., 160.]),
                'Kxd': torch.tensor([35., 35., 35., 14., 14., 14.])
            }
        }

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])

        self.gripper = GripperInterface(
            ip_address="192.168.0.2",
        )

        self.signal_pub = rospy.Publisher('/signal', UInt8, queue_size=10)

        # Reset to rest pose
        self.rest_pos = torch.Tensor(rest_pose[0])
        self.rest_quat = torch.Tensor(rest_pose[1])
        # self.reset()


    def __del__(self):
        self.arm.terminate_current_policy()


    def reset(self, time_to_go=4.0):
        self.move_to(self.rest_pos, self.rest_quat, time_to_go)
        self.open_gripper()


    def update_urdf(self, new_urdf_path):
        with open(new_urdf_path, 'r') as file:
            new_urdf_file = file.read()

        with tempfile.NamedTemporaryFile("w+") as urdf_file:
            urdf_file.write(new_urdf_file)
            self.arm.set_robot_model(urdf_file.name, self.arm.metadata.ee_link_name)


    def set_policy(self, Kx, Kxd):
        torch_policy = toco.policies.CartesianImpedanceControl(
            joint_pos_current=self.arm.get_joint_positions(),
            Kp=Kx,
            Kd=Kxd,
            robot_model=self.arm.robot_model,
            ignore_gravity=self.arm.use_grav_comp,
        )

        return self.arm.send_torch_policy(torch_policy=torch_policy, blocking=False)


    def move_to(self, pos, quat, time_to_go=2.0):
        # print(f'Planned pose: {pos}, {quat}')

        # Plan trajectory
        pos_curr, quat_curr = self.arm.get_ee_pose()
        waypoints = toco.planning.generate_cartesian_space_min_jerk(
            start=T.from_rot_xyz(R.from_quat(quat_curr), pos_curr),
            goal=T.from_rot_xyz(R.from_quat(quat), pos),
            time_to_go=time_to_go,
            hz=self.arm.metadata.hz,
        )

        # Execute trajectory
        for i in range(0, len(waypoints), int(len(waypoints) * planner_dt / time_to_go)):
            # Update traj
            ee_pos_desired = waypoints[i]["pose"].translation()
            ee_quat_desired = waypoints[i]["pose"].rotation().as_quat()

            self.arm.update_desired_ee_pose(ee_pos_desired, ee_quat_desired)

            time.sleep(planner_dt)

        # Wait for robot to stabilize
        time.sleep(0.2)

        pos_actual, quat_actual = self.arm.get_ee_pose()
        # print(f'Actual pose: {pos_actual}, {quat_actual}')
        print(f'Pose error: {pos_actual - pos}, {quat_actual - quat}')


    def close_gripper(self, grasp_width):
        self.gripper.goto(width=grasp_width, speed=grasp_speed, force=grasp_force)
        time.sleep(0.2)

        # Check state
        state = self.gripper.get_state()
        assert state.width < state.max_width


    def open_gripper(self):
        max_width = self.gripper.get_state().max_width
        self.gripper.goto(width=max_width, speed=grasp_speed, force=grasp_force)
        time.sleep(0.2)

        # Check state
        state = self.gripper.get_state()
        assert state.width > 0.0


    def grasp_pose_to_pos_quat(self, grasp_pose, z):
        x, y, rz = grasp_pose
        pos = torch.Tensor([x, y, z])
        quat = (
            R.from_rotvec(torch.Tensor([0, 0, rz])) * R.from_quat(self.rest_quat)
        ).as_quat()

        return pos, quat


    def grasp(self, grasp_params, grasp_h, pregrasp_dh, grasp_width=0.0):
        # Move to pregrasp
        pregrasp_pose = self.grasp_pose_to_pos_quat(grasp_params, grasp_h + pregrasp_dh)
        grasp_pose = self.grasp_pose_to_pos_quat(grasp_params, grasp_h)
        
        print("Pregrasp:")
        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*pregrasp_pose)

        # Lower (slower than other motions to prevent sudden collisions)
        print("Grasp:")
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])
        self.move_to(*grasp_pose)

        self.signal_pub.publish(UInt8(1))
        time.sleep(0.2)

        # Grasp
        self.close_gripper(grasp_width)
        # Release
        self.open_gripper()
        # Lift to pregrasp

        self.signal_pub.publish(UInt8(0))

        print("Back to pregrasp:")
        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*pregrasp_pose)


    def take_away(self, grasp_params, grasp_h, pregrasp_dh, grasp_width=0.02):
        self.open_gripper()

        prep_pose = self.grasp_pose_to_pos_quat((mid_point[0], mid_point[1], grasp_params[2]), grasp_h + pregrasp_dh)
        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*prep_pose)

        # Move to pregrasp
        print("Pregrasp:")
        pregrasp_pose = self.grasp_pose_to_pos_quat(grasp_params, grasp_h + pregrasp_dh)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])
        self.move_to(*pregrasp_pose)

        # Lower (slower than other motions to prevent sudden collisions)
        print("Grasp:")
        grasp_pose = self.grasp_pose_to_pos_quat(grasp_params, grasp_h)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])
        self.move_to(*grasp_pose)

        # Grasp
        self.close_gripper(grasp_width)
        
        # grasp the tool from the shelf
        print("Take away the tool:")
        idle_pose = self.grasp_pose_to_pos_quat((grasp_params[0], grasp_params[1] - 0.06, grasp_params[2]), grasp_h)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])
        self.move_to(*idle_pose)

        prep_pose = self.grasp_pose_to_pos_quat((mid_point[0], mid_point[1], grasp_params[2]), grasp_h + pregrasp_dh)
        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*prep_pose)


    def put_back(self, grasp_params, grasp_h, pregrasp_dh):
        prep_pose = self.grasp_pose_to_pos_quat((mid_point[0], mid_point[1], grasp_params[2]), grasp_h + pregrasp_dh)
        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*prep_pose)

        # Get to the insert position
        print("Get to the insert position:")
        idle_pose = self.grasp_pose_to_pos_quat((grasp_params[0], grasp_params[1] - 0.06, grasp_params[2]), grasp_h)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])
        self.move_to(*idle_pose)

        print("Insert:")
        grasp_pose = self.grasp_pose_to_pos_quat(grasp_params, grasp_h)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])
        self.move_to(*grasp_pose)

        # Release
        self.open_gripper()

        # Move to pregrasp
        print("Lift:")
        pregrasp_pose = self.grasp_pose_to_pos_quat(grasp_params, grasp_h + pregrasp_dh)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])
        self.move_to(*pregrasp_pose)

        prep_pose = self.grasp_pose_to_pos_quat((mid_point[0], mid_point[1], grasp_params[2]), grasp_h + pregrasp_dh)
        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*prep_pose)


    def get_gripper_pose(self):
        ee_pos, ee_quat = self.arm.pose_ee()
        ee_ori = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])
        gripper_width = self.gripper.get_state().width
        print(f"Current ee position: {ee_pos}")
        print(f"Current ee orientation: {ee_ori}  (wxyz)")
        print(f"Current gripper width: {gripper_width}")

        fingertip_pos = (quat2mat(ee_ori) @ ee_fingertip_T_mat[:3, 3].T).T + ee_pos.numpy()
        fingertip_ori = qmult(ee_ori, mat2quat(ee_fingertip_T_mat[:3, :3]))
        print(f"Current fingertip position: {fingertip_pos}")
        print(f"Current fingertip orientation: {fingertip_ori}  (wxyz)")

        ft_gripper_trans_1 = np.array([gripper_ft_trans[0], gripper_ft_trans[1] + gripper_width / 2, gripper_ft_trans[2]])
        ft_gripper_trans_2 = np.array([gripper_ft_trans[0], -gripper_ft_trans[1] - gripper_width / 2, gripper_ft_trans[2]])
        
        gripper_poses = []
        for trans in [ft_gripper_trans_1, ft_gripper_trans_2]:
            gripper_pos = (quat2mat(fingertip_ori) @ trans.T).T + fingertip_pos
            # gripper_ori = fingertip_ori
            gripper_pose = Pose()
            gripper_pose.position.x = gripper_pos[0]
            gripper_pose.position.y = gripper_pos[1]
            gripper_pose.position.z = gripper_pos[2]

            gripper_pose.orientation.w = 1.0
            gripper_pose.orientation.x = 0.0
            gripper_pose.orientation.y = 0.0
            gripper_pose.orientation.z = 0.0
            gripper_poses.append(gripper_pose)

        return gripper_poses
