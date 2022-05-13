import numpy as np
import os
import rospy
import tempfile
import time
import torch
import torchcontrol as toco
import sys
import yaml

torch.set_printoptions(precision=4, profile='short', sci_mode=False)

from geometry_msgs.msg import Pose
from polymetis import RobotInterface, GripperInterface
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from timeit import default_timer as timer
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from transforms3d.quaternions import *


# robot params
ee_fingertip_T_mat = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
gripper_ft_trans = np.array([0.0, 0.019, 0.042192])


class ManipulatorSystem:
    def __init__(self):
        self.arm = RobotInterface(
            ip_address="192.168.0.2",
            enforce_version=False,
        )

        self.gain_dict = {
            'low': {
                'Kx': torch.tensor([300., 300., 300., 40., 40., 40.]),
                'Kxd': torch.tensor([35., 35., 35., 14., 14., 14.])
            },
            'high': {
                'Kx': torch.tensor([1200., 1200., 1200., 160., 160., 160.]),
                'Kxd': torch.tensor([35., 35., 35., 14., 14., 14.])
            }
        }

        self.planner_dt = 0.01

        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        self.gripper = GripperInterface(
            ip_address="192.168.0.2",
        )

        self.signal_pub = rospy.Publisher('/signal', UInt8, queue_size=10)

        # Reset to rest pose
        # self.rest_pose = self.pos_rz_to_pose((0.437, 0.0, 0), 0.4) # for robocraft
        self.rest_pose = self.pos_rz_to_pose((0.4, -0.1, np.pi / 4), 0.4) # for robocook
        self.rest_pos = self.rest_pose[0]
        self.rest_quat = self.rest_pose[1]

        self.grip_speed = 0.1
        self.grip_force = 10.0

        cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        self.tool_status_path = os.path.join(cd, '..', 'env', 'tool_status.yml')
        if os.path.exists(self.tool_status_path):
            with open(self.tool_status_path, 'r') as f:
                self.tool_status = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.tool_status = {
                'circular_cutter': 'ready',
                'gripper': 'ready',
                'hook': 'ready',
                'planar_cutter': 'ready',
                'roller': 'ready',
                'shovel': 'ready',
                'stamp': 'ready',
            }

        # self.reset()


    def __del__(self):
        self.arm.terminate_current_policy()


    def reset(self, time_to_go=3.0):
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
        for i in range(0, len(waypoints), int(len(waypoints) * self.planner_dt / time_to_go)):
            # Update traj
            ee_pos_desired = waypoints[i]["pose"].translation()
            ee_quat_desired = waypoints[i]["pose"].rotation().as_quat()

            self.arm.update_desired_ee_pose(ee_pos_desired, ee_quat_desired)

            time.sleep(self.planner_dt)

        # Wait for robot to stabilize
        time.sleep(0.1)

        pos_actual, quat_actual = self.arm.get_ee_pose()
        # print(f'Actual pose: {pos_actual}, {quat_actual}')
        pos_err = [round(p, 3) for p in (pos_actual - pos).tolist()]
        quat_err = [round(q, 3) for q in (quat_actual - quat).tolist()]
        print(f'\tpose error: {pos_err}, {quat_err}')


    def close_gripper(self, grip_width, blocking=True, grip_params=None):
        if grip_params:
            self.gripper.goto(width=grip_width, speed=grip_params[0], force=grip_params[1], blocking=blocking)
        else:
            self.gripper.goto(width=grip_width, speed=self.grip_speed, force=self.grip_force, blocking=blocking)
        time.sleep(0.2)

        # Check state
        state = self.gripper.get_state()
        assert state.width < state.max_width


    def open_gripper(self):
        max_width = self.gripper.get_state().max_width
        self.gripper.goto(width=max_width, speed=self.grip_speed, force=self.grip_force)
        time.sleep(0.2)

        # Check state
        state = self.gripper.get_state()
        assert state.width > 0.0


    def pos_rz_to_pose(self, grip_pose, z):
        x, y, rz = grip_pose
        pos = torch.Tensor([x, y, z])
        quat = (
            R.from_rotvec(torch.Tensor([0, 0, rz])) * R.from_quat(torch.Tensor([1, 0, 0, 0]))
        ).as_quat()

        return pos, quat


    def pos_rot_to_pose(self, pos, rot):
        pos = torch.Tensor(pos)
        quat = (
            R.from_rotvec(torch.Tensor(rot)) * R.from_quat(torch.Tensor([1, 0, 0, 0]))
        ).as_quat()

        return pos, quat


    def get_ee_pose(self):
        ee_pos, ee_quat = self.arm.get_ee_pose()
        
        ee_pose = Pose()
        ee_pose.position.x = ee_pos[0]
        ee_pose.position.y = ee_pos[1]
        ee_pose.position.z = ee_pos[2]

        ee_pose.orientation.w = ee_quat[3]
        ee_pose.orientation.x = ee_quat[0]
        ee_pose.orientation.y = ee_quat[1]
        ee_pose.orientation.z = ee_quat[2]

        return ee_pose


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


    def take_away_tool(self, tool):
        if tool == 'gripper':
            self.take_away(grip_params=(0.47, 0.26, np.pi / 4), grip_h=0.315, pregrip_dh=0.01, grip_width=0.01, lift_dh=0.1, loc='left')
        elif tool == 'roller':
            self.take_away(grip_params=(0.66, 0.185, -np.pi / 4), grip_h=0.325, pregrip_dh=0.01, grip_width=0.015)
        elif tool == 'planar_cutter':
            self.take_away(grip_params=(0.66, 0.065, -np.pi / 4), grip_h=0.325, pregrip_dh=0.01, grip_width=0.015)
        elif tool == 'stamp':
            self.take_away(grip_params=(0.66, -0.1, -np.pi / 4), grip_h=0.325, pregrip_dh=0.01, grip_width=0.015)
        elif tool == 'shovel':
            self.take_away(grip_params=(0.66, -0.225, -np.pi / 4), grip_h=0.325, pregrip_dh=0.01, grip_width=0.015)
        else:
            raise NotImplementedError

        with open(self.tool_status_path, 'w') as f:
            yaml.dump(self.tool_status, f)


    def put_back_tool(self, tool):
        if tool == 'gripper':
            self.put_back_gripper(grip_params=(0.467, 0.26, np.pi / 4), grip_h=0.315, pregrip_dh=0.03)
        elif tool == 'roller':
            self.put_back(grip_params=(0.655, 0.19, -np.pi / 4), grip_h=0.325, pregrip_dh=0.015)
        elif tool == 'planar_cutter':
            self.put_back(grip_params=(0.655, 0.065, -np.pi / 4), grip_h=0.325, pregrip_dh=0.015)
        elif tool == 'stamp':
            self.put_back(grip_params=(0.655, -0.1, -np.pi / 4), grip_h=0.325, pregrip_dh=0.015)
        elif tool == 'shovel':
            self.put_back(grip_params=(0.655, -0.225, -np.pi / 4), grip_h=0.325, pregrip_dh=0.015)
        else:
            raise NotImplementedError

        with open(self.tool_status_path, 'w') as f:
            yaml.dump(self.tool_status, f)


    def take_away(self, grip_params, grip_h, pregrip_dh, grip_width, lift_dh=0.01, loc='front', debug=False):
        pregrip_h = grip_h + pregrip_dh
        lift_h = grip_h + pregrip_dh + lift_dh

        # hard code to account for taking away the gripper
        if loc == 'left':
            self.open_gripper()
        else:
            self.close_gripper(grip_width + 0.04, blocking=False)

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*self.rest_pose)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        prep_pose = self.pos_rz_to_pose((self.rest_pos[0], self.rest_pos[1], grip_params[2]), pregrip_h)
        self.move_to(*prep_pose)

        # Move to an idle pose before pregrip
        print("=> align two axes:")
        if loc == 'front':
            prepregrip_pose = self.pos_rz_to_pose((grip_params[0] - 0.1, grip_params[1], grip_params[2]), pregrip_h)
        elif loc == 'left':
            prepregrip_pose = self.pos_rz_to_pose((grip_params[0], grip_params[1] - 0.1, grip_params[2]), pregrip_h)
        elif loc == 'right':
            prepregrip_pose = self.pos_rz_to_pose((grip_params[0], grip_params[1] + 0.1, grip_params[2]), pregrip_h)
        else:
            raise NotImplementedError
        self.move_to(*prepregrip_pose)

        # Move to pregrip
        print("=> move to pregrip:")
        pregrip_pose = self.pos_rz_to_pose(grip_params, pregrip_h)
        self.move_to(*pregrip_pose)

        # Lower (slower than other motions to prevent sudden collisions)
        print("=> grip:")
        grip_pose = self.pos_rz_to_pose(grip_params, grip_h)
        self.move_to(*grip_pose)

        # grip
        if not debug:
            self.close_gripper(grip_width, blocking=False)
            time.sleep(0.2)

        # Lift the tool
        print("=> lift the tool:")
        lift_pose = self.pos_rz_to_pose(grip_params, lift_h)
        self.move_to(*lift_pose)
        
        # grip the tool from the shelf
        print("=> take away the tool:")
        if loc == 'front':
            idle_pose = self.pos_rz_to_pose((grip_params[0] - 0.1, grip_params[1], grip_params[2]), lift_h)
        elif loc == 'left':
            idle_pose = self.pos_rz_to_pose((grip_params[0], grip_params[1] - 0.1, grip_params[2]), lift_h)
        elif loc == 'right':
            idle_pose = self.pos_rz_to_pose((grip_params[0], grip_params[1] + 0.1, grip_params[2]), lift_h)
        else:
            raise NotImplementedError
        self.move_to(*idle_pose)

        print("=> move back to prep pose:")
        prep_pose = self.pos_rz_to_pose((self.rest_pos[0], self.rest_pos[1], grip_params[2]), pregrip_h)
        self.move_to(*prep_pose)

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*self.rest_pose)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])


    def put_back(self, grip_params, grip_h, pregrip_dh, lift_dh=0.1, loc='front'):
        pregrip_h = grip_h + pregrip_dh
        lift_h = grip_h + pregrip_dh + lift_dh

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*self.rest_pose)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        prep_pose = self.pos_rz_to_pose((self.rest_pos[0], self.rest_pos[1], grip_params[2]), pregrip_h)
        self.move_to(*prep_pose)

        # Get to the insert position
        print("=> get to the insert position:")
        if loc == 'front':
            preinsert_pose = self.pos_rz_to_pose((grip_params[0] - 0.1, grip_params[1], grip_params[2]), pregrip_h)
        elif loc == 'left':
            preinsert_pose = self.pos_rz_to_pose((grip_params[0], grip_params[1] - 0.1, grip_params[2]), pregrip_h)
        elif loc == 'right':
            preinsert_pose = self.pos_rz_to_pose((grip_params[0], grip_params[1] + 0.1, grip_params[2]), pregrip_h)
        else:
            raise NotImplementedError
        self.move_to(*preinsert_pose)

        print("=> insert:")
        insert_pose = self.pos_rz_to_pose(grip_params, pregrip_h)
        self.move_to(*insert_pose)

        # Release
        self.open_gripper()

        # Move to pregrip
        print("=> lift:")
        lift_pose = self.pos_rz_to_pose(grip_params, lift_h)
        self.move_to(*lift_pose)

        print("=> move back to prep pose:")
        prep_pose = self.pos_rz_to_pose((self.rest_pos[0], self.rest_pos[1], grip_params[2]), pregrip_h)
        self.move_to(*prep_pose)

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*self.rest_pose)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])


    def put_back_gripper(self, grip_params, grip_h, pregrip_dh, lift_dh=0.1):
        pregrip_h = grip_h + pregrip_dh
        lift_h = grip_h + pregrip_dh + lift_dh

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*self.rest_pose)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        prep_pose = self.pos_rz_to_pose((self.rest_pos[0], self.rest_pos[1], grip_params[2]), pregrip_h)
        self.move_to(*prep_pose)

        self.close_gripper(0.04, blocking=False)

         # Get to the insert position
        print("=> get to the insert position:")
        preinsert_pose = self.pos_rz_to_pose((grip_params[0], grip_params[1] - 0.1, grip_params[2]), pregrip_h)
        self.move_to(*preinsert_pose)

        # Get to the put-down position
        print("=> get to the put-down position:")
        idle_pose = self.pos_rz_to_pose(grip_params, pregrip_h)
        self.move_to(*idle_pose)

        print("=> put down:")
        grip_pose = self.pos_rz_to_pose(grip_params, grip_h)
        self.move_to(*grip_pose)

        # Release
        self.open_gripper()

        # Move to pregrip
        print("=> lift:")
        pregrip_pose = self.pos_rz_to_pose(grip_params, lift_h)
        self.move_to(*pregrip_pose)

        print("=> back to prep pose:")
        prep_pose = self.pos_rz_to_pose((self.rest_pos[0], self.rest_pos[1], grip_params[2]), pregrip_h)
        self.move_to(*prep_pose)

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*self.rest_pose)
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])


    def cut(self, cut_pos, cut_rot, precut_dh, mode='explore'):
        # Move to precut
        print("=> precut:")
        precut_pos = [cut_pos[0], cut_pos[1], cut_pos[2] + precut_dh]
        precut_pose = self.pos_rot_to_pose(precut_pos, cut_rot)
        self.move_to(*precut_pose)

        if mode == 'explore':
            self.signal_pub.publish(UInt8(1))
            time.sleep(0.2)

        # Cut
        print("=> cut:")
        cut_pose = self.pos_rot_to_pose(cut_pos, cut_rot)
        self.move_to(*cut_pose)

        # Separate
        print("=> separate:")
        separate_pos = [cut_pos[0], cut_pos[1] + 0.05, cut_pos[2]]
        separate_pose = self.pos_rot_to_pose(separate_pos, cut_rot)
        self.move_to(*separate_pose)

        if mode == 'explore':
            self.signal_pub.publish(UInt8(0))
            time.sleep(0.2)

        print("=> back to precut:")
        self.move_to(*precut_pose)


    def grip(self, grip_params, grip_h, pregrip_dh, grip_width, mode='explore'):
        # Move to pregrip
        print("=> pregrip:")
        self.open_gripper()
        pregrip_pose = self.pos_rz_to_pose(grip_params, grip_h + pregrip_dh)
        self.move_to(*pregrip_pose)

        # Lower
        print("=> grip:")
        grip_pose = self.pos_rz_to_pose(grip_params, grip_h)
        self.move_to(*grip_pose, time_to_go=1.0)

        if mode == 'explore':
            self.signal_pub.publish(UInt8(1))
            time.sleep(0.1)

        # grip
        self.close_gripper(grip_width, blocking=False, grip_params=(0.02, 150))

        # Release
        self.open_gripper()
        # Lift to pregrip
        
        if mode == 'explore':
            self.signal_pub.publish(UInt8(0))
            time.sleep(0.1)

        print("=> back to pregrip:")
        self.move_to(*pregrip_pose)


    def roll(self, start_pos, roll_rot, end_pos, preroll_dh, mode='explore'):
        # Move to preroll
        print("=> preroll:")
        preroll_pos = [start_pos[0], start_pos[1], start_pos[2] + preroll_dh]
        preroll_pose = self.pos_rot_to_pose(preroll_pos, roll_rot)
        self.move_to(*preroll_pose)

        if mode == 'explore':
            self.signal_pub.publish(UInt8(1))
            time.sleep(0.1)

        # Press
        print("=> press:")
        press_pose = self.pos_rot_to_pose(start_pos, roll_rot)
        self.move_to(*press_pose)

        # Spread
        print("=> spread:")
        spread_pose = self.pos_rot_to_pose(end_pos, roll_rot)
        self.move_to(*spread_pose)

        if mode == 'explore':
            self.signal_pub.publish(UInt8(0))
            time.sleep(0.1)

        print("=> back to preroll:")
        self.move_to(*preroll_pose)


    def press(self, press_pos, press_rot, prepress_dh, mode='explore'):
        # Move to prepress
        print("=> prepress:")
        prepress_pos = [press_pos[0], press_pos[1], press_pos[2] + prepress_dh]
        prepress_pose = self.pos_rot_to_pose(prepress_pos, press_rot)
        self.move_to(*prepress_pose)

        if mode == 'explore':
            self.signal_pub.publish(UInt8(1))
            time.sleep(0.2)

        # Press
        print("=> press:")
        press_pose = self.pos_rot_to_pose(press_pos, press_rot)
        self.move_to(*press_pose)

        if mode == 'explore':
            self.signal_pub.publish(UInt8(0))
            time.sleep(0.2)

        print("=> back to prepress:")
        self.move_to(*prepress_pose)
