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
from transforms3d.axangles import axangle2mat
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
            'medium': {
                'Kx': torch.tensor([1200., 1200., 1200., 160., 160., 160.]),
                'Kxd': torch.tensor([35., 35., 35., 14., 14., 14.])
            },
            'high': {
                'Kx': torch.tensor([2400., 2400., 2400., 640., 640., 640.]),
                'Kxd': torch.tensor([35., 35., 35., 14., 14., 14.])
            }
        }

        self.planner_dt = 0.01

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])

        self.gripper = GripperInterface(
            ip_address="192.168.0.2",
        )

        # 0: start, 1: end, 2: away
        self.action_signal_pub = rospy.Publisher('/action_signal', UInt8, queue_size=10)

        # Reset to rest pose
        # self.rest_pose = self.pos_rz_to_pose((0.437, 0.0, 0), 0.4) # for robocraft
        self.rest_pose = self.pos_rz_to_pose((0.4, -0.1, np.pi / 4), 0.4) # for robocook
        self.rest_pos = self.rest_pose[0]
        self.rest_quat = self.rest_pose[1]

        self.grip_speed = 0.05
        self.grip_force = 20.0

        cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        self.tool_status_path = os.path.join(cd, '..', 'env', 'tool_status.yml')
        if os.path.exists(self.tool_status_path):
            with open(self.tool_status_path, 'r') as f:
                self.tool_status = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.tool_status = {
                'gripper_asym': {'loc': 'l0-0', 'status': 'ready'},
                'gripper_sym_plane': {'loc': 'l0-1', 'status': 'ready'},
                'gripper_sym_rod': {'loc': 'l0-2', 'status': 'ready'},
                'spatula_small': {'loc': 'l1-0', 'status': 'ready'},
                'spatula_large': {'loc': 'l1-1', 'status': 'ready'},
                'roller_large': {'loc': 'f0-low', 'status': 'ready'},
                'press_circle': {'loc': 'f0-mid', 'status': 'ready'},
                'punch_circle': {'loc': 'f0-high', 'status': 'ready'},
                'roller_small': {'loc': 'f1-low', 'status': 'ready'},
                'press_square': {'loc': 'f1-mid', 'status': 'ready'},
                'punch_square': {'loc': 'f1-high', 'status': 'ready'},
                'cutter_planar': {'loc': 'f2-low', 'status': 'ready'},
                'cutter_circular': {'loc': 'f2-mid', 'status': 'ready'},
                'hook': {'loc': 'f2-high', 'status': 'ready'},
            }

        # with open(self.tool_status_path, 'w') as f:
        #     yaml.dump(self.tool_status, f)

        self.loc_param_dict = {
            'l0-0': [0.3425, 0.29, -np.pi / 4, 0.32],
            'l0-1': [0.4125, 0.29, -np.pi / 4, 0.32],
            'l0-2': [0.4625, 0.29, -np.pi / 4, 0.32],
            'l1-0': [0.6225, 0.2065, np.pi / 4, 0.325],
            'l1-1': [0.6225, 0.2815, np.pi / 4, 0.325],
            'f0-low': [0.5975, -0.265, -np.pi / 4, 0.225],
            'f0-mid': [0.6575, -0.27, -np.pi / 4, 0.3325],
            'f0-high': [0.725, -0.2725, -np.pi / 4, 0.44],
            'f1-low': [0.6025, -0.115, -np.pi / 4, 0.225],
            'f1-mid': [0.665, -0.115, -np.pi / 4, 0.3325],
            'f1-high': [0.7275, -0.1175, -np.pi / 4, 0.44],
            'f2-low': [0.6025, 0.04, -np.pi / 4, 0.225],
            'f2-mid': [0.665, 0.0375, -np.pi / 4, 0.3325],
            'f2-high': [0.7275, 0.04, -np.pi / 4, 0.44],
        }

        # self.reset()


    def __del__(self):
        self.arm.terminate_current_policy()


    def reset(self, time_to_go=2.0):
        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(self.rest_pos, self.rest_quat, time_to_go)


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


    def publish(self, signal):
        self.action_signal_pub.publish(UInt8(signal))
        time.sleep(0.2)


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


    def open_gripper(self, blocking=True, grip_params=None):
        max_width = self.gripper.get_state().max_width
        if grip_params:
            self.gripper.goto(width=max_width, speed=grip_params[0], force=grip_params[1], blocking=blocking)
        else:
            self.gripper.goto(width=max_width, speed=self.grip_speed, force=self.grip_force, blocking=blocking)

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


    def pos_mat_to_pose(self, pos, rot):
        pos = torch.Tensor(pos)
        quat = (
            R.from_matrix(torch.Tensor(rot)) * R.from_quat(torch.Tensor([1, 0, 0, 0]))
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


    def take_away_tool(self, tool, debug=False):
        if tool in self.tool_status.keys():
            loc = self.tool_status[tool]['loc']
        else:
            raise NotImplementedError

        if 'l0' in loc or 'l1' in loc:
            self.take_away(grip_params=self.loc_param_dict[loc][:3], 
                grip_h=self.loc_param_dict[loc][3], pregrip_dh=0.015,
                grip_width=0.01, lift_dh=0.15, loc=loc, debug=debug)
        else:
            self.take_away(grip_params=self.loc_param_dict[loc][:3], 
                grip_h=self.loc_param_dict[loc][3], pregrip_dh=0.015, 
                grip_width=0.0165, lift_dh=0.005, loc=loc, debug=debug)

        if not debug:
            self.tool_status[tool]['status'] = 'using'
            with open(self.tool_status_path, 'w') as f:
                yaml.dump(self.tool_status, f)


    def put_back_tool(self, tool):
        if tool in self.tool_status.keys():
            loc = self.tool_status[tool]['loc']
        else:
            raise NotImplementedError

        if 'l0' in loc or 'l1' in loc:
            self.put_back(grip_params=self.loc_param_dict[loc][:3],
                grip_h=self.loc_param_dict[loc][3] - 0.005, 
                pregrip_dh=0.03, lift_dh=0.15, loc=loc)
        else:
            self.put_back(grip_params=self.loc_param_dict[loc][:3], 
                grip_h=self.loc_param_dict[loc][3] - 0.01, 
                pregrip_dh=0.015, lift_dh=0.07, loc=loc)

        self.tool_status[tool]['status'] = 'ready'
        with open(self.tool_status_path, 'w') as f:
            yaml.dump(self.tool_status, f)


    def take_away(self, grip_params, grip_h, pregrip_dh, grip_width, lift_dh, loc, debug=False):
        pregrip_h = grip_h + pregrip_dh
        lift_h = grip_h + lift_dh

        if 'l0' in loc or 'l1' in loc:
            self.open_gripper()
        else:
            self.close_gripper(grip_width + 0.03, blocking=False)

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*self.rest_pose, time_to_go=2.0)
        self.set_policy(self.gain_dict['medium']['Kx'], self.gain_dict['medium']['Kxd'])
        # prep_pose = self.pos_rz_to_pose((self.rest_pos[0], self.rest_pos[1], grip_params[2]), pregrip_h)
        # self.move_to(*prep_pose, time_to_go=1.0)

        if 'l0' in loc or 'l1' in loc:
            if 'l0' in loc:
                prep_params = (grip_params[0] + 0.02, grip_params[1], grip_params[2])
            else:
                prep_params = (grip_params[0], grip_params[1] - 0.02, grip_params[2])

            print("=> align y-axis:")
            preprepregrip_pose = self.pos_rz_to_pose(prep_params, lift_h)
            self.move_to(*preprepregrip_pose, time_to_go=2.0)

            print("=> move down:")
            prepregrip_pose = self.pos_rz_to_pose(prep_params, pregrip_h)
            self.move_to(*prepregrip_pose, time_to_go=2.0)

            print("=> move to pregrip:")
            pregrip_pose = self.pos_rz_to_pose(grip_params, pregrip_h)
            self.move_to(*pregrip_pose, time_to_go=2.0)
        else:
            print("=> align two axes:")
            prepregrip_pose = self.pos_rz_to_pose((grip_params[0] - 0.1, grip_params[1], grip_params[2]), pregrip_h)
            self.move_to(*prepregrip_pose, time_to_go=2.0)

            print("=> move to pregrip:")
            pregrip_pose = self.pos_rz_to_pose(grip_params, pregrip_h)
            self.move_to(*pregrip_pose, time_to_go=2.0)

        print("=> grip:")
        grip_pose = self.pos_rz_to_pose(grip_params, grip_h)
        self.move_to(*grip_pose, time_to_go=2.0)

        # grip
        if not debug:
            if 'l1-1' in loc:
                self.close_gripper(0.03, blocking=False)
                time.sleep(1.5)
            else:
                self.close_gripper(grip_width, blocking=False)
                time.sleep(0.5)

        # Lift the tool
        print("=> lift the tool:")
        lift_pose = self.pos_rz_to_pose(grip_params, lift_h)
        if 'l0' in loc or 'l1' in loc:
            self.move_to(*lift_pose, time_to_go=2.0)
        else:
            self.move_to(*lift_pose, time_to_go=1.0)
        
        # grip the tool from the shelf
        print("=> take away the tool:")
        if not 'l0' in loc and not 'l1' in loc:
            idle_pose = self.pos_rz_to_pose((grip_params[0] - 0.1, grip_params[1], grip_params[2]), lift_h)
            self.move_to(*idle_pose, time_to_go=2.0)

        # print("=> move back to prep pose:")
        # prep_pose = self.pos_rz_to_pose((self.rest_pos[0], self.rest_pos[1], grip_params[2]), pregrip_h)
        # self.move_to(*prep_pose)

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*self.rest_pose, time_to_go=2.0)


    def put_back(self, grip_params, grip_h, pregrip_dh, lift_dh, loc):
        pregrip_h = grip_h + pregrip_dh
        lift_h = grip_h + lift_dh

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*self.rest_pose, time_to_go=2.0)
        self.set_policy(self.gain_dict['medium']['Kx'], self.gain_dict['medium']['Kxd'])

        # prep_pose = self.pos_rz_to_pose((self.rest_pos[0], self.rest_pos[1], grip_params[2]), pregrip_h)
        # self.move_to(*prep_pose, time_to_go=1.0)

        if 'l0' in loc or 'l1' in loc:
            close_width = 0.04 if 'l1' in loc else 0.03
            self.close_gripper(close_width, blocking=False)

        if 'l0' in loc or 'l1' in loc:
            print("=> align y-axis:")
            preinsert_pose = self.pos_rz_to_pose(grip_params, lift_h)
            self.move_to(*preinsert_pose, time_to_go=2.0)

            # print("=> get to the insert position:")
            # insert_pose = self.pos_rz_to_pose((grip_params[0] + 0.005, grip_params[1], grip_params[2]), pregrip_h)
            # self.move_to(*insert_pose)

            # print("=> get to the put-down position:")
            # idle_pose = self.pos_rz_to_pose(grip_params, pregrip_h)
            # self.move_to(*idle_pose, time_to_go=0.5)

            print("=> put down:")
            grip_pose = self.pos_rz_to_pose(grip_params, grip_h)
            self.move_to(*grip_pose, time_to_go=2.0)
        else:
            print("=> get to the insert position:")
            preinsert_pose = self.pos_rz_to_pose((grip_params[0] - 0.1, grip_params[1], grip_params[2]), pregrip_h)
            self.move_to(*preinsert_pose, time_to_go=2.0)

            print("=> insert:")
            insert_pose = self.pos_rz_to_pose(grip_params, pregrip_h)
            self.move_to(*insert_pose, time_to_go=2.0)

        # Release
        self.open_gripper()
        time.sleep(0.5)

        # Move to pregrip
        print("=> lift:")
        lift_pose = self.pos_rz_to_pose(grip_params, lift_h)
        self.move_to(*lift_pose, time_to_go=2.0)

        # print("=> move back to prep pose:")
        # prep_pose = self.pos_rz_to_pose((self.rest_pos[0], self.rest_pos[1], grip_params[2]), pregrip_h)
        # self.move_to(*prep_pose)

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        self.move_to(*self.rest_pose, time_to_go=2.0)


    def grip(self, grip_params, grip_h, pregrip_dh, grip_width):
        self.publish(0)

        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        # Move to pregrip
        print("=> pregrip:")
        self.open_gripper()
        pregrip_pose = self.pos_rz_to_pose(grip_params, grip_h + pregrip_dh)
        self.move_to(*pregrip_pose)

        # Lower
        print("=> grip:")
        grip_pose = self.pos_rz_to_pose(grip_params, grip_h)
        self.move_to(*grip_pose, time_to_go=1.0)

        self.publish(1)

        # grip
        self.close_gripper(grip_width, blocking=False, grip_params=(0.01, 300))

        # if mode == 'react':
        #     time.sleep(5.0)
        #     self.action_signal_pub.publish(UInt8(1))
        #     time.sleep(1.0)

        # Release
        self.open_gripper()
        # Lift to pregrip

        self.publish(2)

        print("=> back to pregrip:")
        self.move_to(*pregrip_pose)

        self.reset()
        self.publish(3)


    def press(self, press_pos, press_rot, prepress_dh, mode='explore'):
        self.publish(0)
        
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        # Move to prepress
        print("=> prepress:")
        prepress_pos = [press_pos[0], press_pos[1], press_pos[2] + prepress_dh]
        prepress_pose = self.pos_rot_to_pose(prepress_pos, press_rot)
        self.move_to(*prepress_pose)

        self.publish(1)

        # Press
        print("=> press:")
        press_pose = self.pos_rot_to_pose(press_pos, press_rot)
        self.move_to(*press_pose)

        self.publish(2)

        print("=> back to prepress:")
        self.move_to(*prepress_pose)

        self.reset()
        self.publish(3)


    def roll(self, start_pos, roll_rot, end_pos, preroll_dh, mode='explore'):
        self.publish(0)

        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        # Move to preroll
        print("=> preroll:")
        preroll_pos = [start_pos[0], start_pos[1], start_pos[2] + preroll_dh]
        preroll_pose = self.pos_rot_to_pose(preroll_pos, roll_rot)
        self.move_to(*preroll_pose)

        self.publish(1)

        # Press
        print("=> press:")
        press_pose = self.pos_rot_to_pose(start_pos, roll_rot)
        self.move_to(*press_pose)

        # Spread
        print("=> spread:")
        spread_pose = self.pos_rot_to_pose(end_pos, roll_rot)
        self.move_to(*spread_pose)

        self.publish(2)
        
        print("=> back to preroll:")
        self.move_to(*preroll_pose)

        self.reset()
        self.publish(3)


    def cut_planar(self, cut_pos, cut_rot, precut_dh, push_y):
        self.publish(0)

        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        # Move to precut
        print("=> precut:")
        precut_pos = [cut_pos[0], cut_pos[1], cut_pos[2] + precut_dh]
        precut_pose = self.pos_rot_to_pose(precut_pos, cut_rot)
        self.move_to(*precut_pose)

        # Cut
        print("=> cut:")
        cut_pose = self.pos_rot_to_pose(cut_pos, cut_rot)
        self.move_to(*cut_pose, time_to_go=3.0)

        # Separate
        print("=> separate:")
        separate_pos = [cut_pos[0], cut_pos[1] + push_y, cut_pos[2]]
        separate_pose = self.pos_rot_to_pose(separate_pos, cut_rot)
        self.move_to(*separate_pose, time_to_go=3.0)

        print("=> back to precut:")
        self.move_to(*precut_pose)

        self.reset()
        self.publish(3)


    def cut_circular(self, cut_pos, cut_rot, precut_dh):
        self.publish(0)

        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        # Move to precut
        print("=> precut:")
        precut_pos = [cut_pos[0], cut_pos[1], cut_pos[2] + precut_dh]
        precut_pose = self.pos_rot_to_pose(precut_pos, cut_rot)
        self.move_to(*precut_pose)

        # Cut
        print("=> cut:")
        cut_pose = self.pos_rot_to_pose(cut_pos, cut_rot)
        self.move_to(*cut_pose, time_to_go=3.0)

        # Rotate around
        cut_rot_x_pos = [0.0, 0.05, 0]
        print("=> rotate to x+:")
        rotate_pose = self.pos_rot_to_pose(cut_pos, cut_rot_x_pos)
        self.move_to(*rotate_pose, time_to_go=1.0)

        cut_rot_x_neg = [0.0, -0.05, 0]
        print("=> rotate to x-:")
        rotate_pose = self.pos_rot_to_pose(cut_pos, cut_rot_x_neg)
        self.move_to(*rotate_pose, time_to_go=1.0)

        cut_rot_y_pos = [0.05, 0.0, 0]
        print("=> rotate to y+:")
        rotate_pose = self.pos_rot_to_pose(cut_pos, cut_rot_y_pos)
        self.move_to(*rotate_pose, time_to_go=1.0)

        cut_rot_y_neg = [-0.05, 0.0, 0]
        print("=> rotate to y-:")
        rotate_pose = self.pos_rot_to_pose(cut_pos, cut_rot_y_neg)
        self.move_to(*rotate_pose, time_to_go=1.0)

        print("=> back to precut:")
        self.move_to(*precut_pose)

        self.reset()
        self.publish(3)


    def push(self, push_center, prepush_dh):
        self.publish(0)
        
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        r = 0.0375
        # moves = [[(-(r+0.02)*np.sqrt(2), 0.02*np.sqrt(2)), (0, (r+0.04)*np.sqrt(2)), 0],
        #     [(-(r+0.02)*np.sqrt(2), -0.02*np.sqrt(2)), (0, -(r+0.04)*np.sqrt(2)), np.pi/2],
        #     [(r+0.04, -0.1), (r+0.04, 0.1), np.pi/4]]
        
        # cut and push vertically, then push horizontally
        moves = [[(0.005, r+0.04+0.002), (r+0.04, r+0.04), 3*np.pi/4],
            [(0.01, r+0.04+0.002), (-r-0.015, r+0.04), 3*np.pi/4],
            [(-r, r+0.04), (-r, -r-0.02), 4*np.pi/5],
            # [(0.005, -r-0.04-0.0025), (-r-0.015, -r-0.04), 3*np.pi/4],
            [(-r-0.015, -r-0.04-0.002), (r+0.04, -r-0.04-0.002), 3*np.pi/4],
            [(r+0.04+0.005, -r-0.1), (r+0.04+0.005, r+0.12), np.pi/4]]

        for i in range(len(moves)):
            # Move to precut
            print(f"=> push:")
            push_rot = [0.0, 0.0, moves[i][2]]
            prepress_pos = [push_center[0] + moves[i][0][0], push_center[1] + moves[i][0][1], push_center[2] + prepush_dh]
            prepress_pose = self.pos_rot_to_pose(prepress_pos, push_rot)
            self.move_to(*prepress_pose)

            prepush_pos = [push_center[0] + moves[i][0][0], push_center[1] + moves[i][0][1], push_center[2] - 0.005]
            # print(prepush_pos)
            prepush_pose = self.pos_rot_to_pose(prepush_pos, push_rot)
            self.move_to(*prepush_pose)
            
            push_pos = [push_center[0] + moves[i][1][0], push_center[1] + moves[i][1][1], push_center[2]]
            # print(push_pos)
            push_pose = self.pos_rot_to_pose(push_pos, push_rot)
            
            time_to_go = 4.0 if i > 3 else 2.0
            self.move_to(*push_pose, time_to_go=time_to_go)

        self.reset()
        self.publish(3)


    def pick_and_place_skin(self, pick_pos, pick_rot, prepick_dh, place_pos, place_rot, preplace_dh, grip_width):
        self.publish(0)

        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        print("=> prepick:")
        self.open_gripper()
        prepick_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + prepick_dh]
        prepick_pose = self.pos_rot_to_pose(prepick_pos, pick_rot)
        self.move_to(*prepick_pose)

        # Lower
        print("=> pick:")
        tuned_pick_rot = axangle2mat([0, 1, 0], -0.01) @ axangle2mat([0, 0, 1], pick_rot[2])
        pick_pose = self.pos_mat_to_pose(pick_pos, tuned_pick_rot)
        self.move_to(*pick_pose, time_to_go=2.0)

        self.close_gripper(grip_width, blocking=False, grip_params=(0.02, 50))
        time.sleep(4.0)

        print("=> back to prepick:")
        self.move_to(*prepick_pose)

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        print("=> prerotate:")
        prerotate_pos = [place_pos[0], place_pos[1] + 0.16, place_pos[2]]
        prerotate_rot = axangle2mat([1, 0, 0], -np.pi / 3) @ axangle2mat([0, 0, 1], np.pi / 4)
        prerotate_pose = self.pos_mat_to_pose(prerotate_pos, prerotate_rot)
        self.move_to(*prerotate_pose, time_to_go=3.0)

        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])
        print("=> preplace:")
        preplace_pos = [place_pos[0], place_pos[1] + 0.18, place_pos[2] - 0.0675]
        preplace_pose = self.pos_mat_to_pose(preplace_pos, prerotate_rot)
        self.move_to(*preplace_pose, time_to_go=3.0)

        # print("=> rotate:")
        # rotate_pos = [place_pos[0], place_pos[1] + 0.2075, place_pos[2] - 0.1225]
        # rotate_rot = axangle2mat([1, 0, 0], -np.pi / 2.25) @ axangle2mat([0, 0, 1], np.pi / 4)
        # rotate_pose = self.pos_mat_to_pose(rotate_pos, rotate_rot)
        # self.move_to(*rotate_pose, time_to_go=6.0)

        self.close_gripper(0.03, blocking=False, grip_params=(0.01, 50))

        print("=> place:")
        down_pos = [place_pos[0], place_pos[1] + 0.1, place_pos[2] - 0.05]
        down_pose = self.pos_mat_to_pose(down_pos, prerotate_rot)
        self.move_to(*down_pose, time_to_go=3.0)

        self.set_policy(self.gain_dict['low']['Kx'], self.gain_dict['low']['Kxd'])
        print("=> back to postplace:")
        postplace_pos = [place_pos[0], place_pos[1], place_pos[2] + preplace_dh]
        postplace_pose = self.pos_rot_to_pose(postplace_pos, place_rot)
        self.move_to(*postplace_pose, time_to_go=6.0)

        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])
        print("=> back to prepick:")
        self.move_to(*prepick_pose)

        self.reset()
        self.publish(3)


    def pick_and_place_filling(self, pick_pos, pick_rot, prepick_dh, place_pos, place_rot, preplace_dh, grip_width):
        self.publish(0)

        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        print("=> prepick:")
        self.open_gripper()
        prepick_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + prepick_dh]
        prepick_pose = self.pos_rot_to_pose(prepick_pos, pick_rot)
        self.move_to(*prepick_pose)

        # Lower
        print("=> pick:")
        pick_pose = self.pos_rot_to_pose(pick_pos, pick_rot)
        self.move_to(*pick_pose, time_to_go=2.0)

        self.close_gripper(grip_width, blocking=False, grip_params=(0.02, 50))
        time.sleep(4.0)

        print("=> back to prepick:")
        self.move_to(*prepick_pose)

        print("=> preplace:")
        preplace_pos = [place_pos[0], place_pos[1], place_pos[2] + preplace_dh]
        preplace_pose = self.pos_rot_to_pose(preplace_pos, place_rot)
        self.move_to(*preplace_pose)

        print("=> place:")
        place_pose = self.pos_rot_to_pose(place_pos, place_rot)
        self.move_to(*place_pose)

        self.open_gripper(grip_params=(0.1, 20))
        
        print("=> back to preplace:")
        self.move_to(*preplace_pose)

        print("=> back to prepick:")
        self.move_to(*prepick_pose)

        self.reset()
        self.publish(3)


    def hook_dumpling_clip(self, hook_pos, debug=False):
        self.publish(0)
        
        self.set_policy(self.gain_dict['high']['Kx'], self.gain_dict['high']['Kxd'])

        hook_rot = [0.0, 0.0, np.pi / 4]

        print("=> prehook:")
        prehook_pose = self.pos_rot_to_pose(hook_pos, hook_rot)
        self.move_to(*prehook_pose)

        print("=> move down:")
        hook_down_pose = self.pos_rot_to_pose(
            (hook_pos[0], hook_pos[1], hook_pos[2] - 0.01), hook_rot)
        self.move_to(*hook_down_pose)

        print("=> move left:")
        hook_left_pose = self.pos_rot_to_pose(
            (hook_pos[0], hook_pos[1] - 0.02, hook_pos[2] - 0.01), hook_rot)
        self.move_to(*hook_left_pose)

        if debug: return

        print("=> move to the highest point:")
        hook_high_pose = self.pos_rot_to_pose(
            (hook_pos[0], hook_pos[1] - 0.03, hook_pos[2] + 0.05), hook_rot)
        self.move_to(*hook_high_pose)

        hook_left_pose = self.pos_rot_to_pose(
            (hook_pos[0], hook_pos[1] - 0.1, hook_pos[2] + 0.06), hook_rot)
        self.move_to(*hook_left_pose)

        print("=> move right a little bit:")
        hook_right_pose = self.pos_rot_to_pose(
            (hook_pos[0], hook_pos[1] - 0.07, hook_pos[2] + 0.07), hook_rot)
        self.move_to(*hook_right_pose)

        print("=> move to the top:")
        hook_top_pose = self.pos_rot_to_pose(
            (hook_pos[0], hook_pos[1] - 0.07, hook_pos[2] + 0.09), hook_rot)
        self.move_to(*hook_top_pose)

        print("=> move left a little bit:")
        hook_left_pose = self.pos_rot_to_pose(
            (hook_pos[0], hook_pos[1] - 0.09, hook_pos[2] + 0.09), hook_rot)
        self.move_to(*hook_left_pose)

        print("=> push down:")
        hook_down_pose = self.pos_rot_to_pose(
            (hook_pos[0], hook_pos[1] - 0.15, hook_pos[2] + 0.025), hook_rot)
        self.move_to(*hook_down_pose)

        print("=> move up a little bit:")
        hook_up_pose = self.pos_rot_to_pose(
            (hook_pos[0], hook_pos[1] - 0.15, hook_pos[2] + 0.04), hook_rot)
        self.move_to(*hook_up_pose)

        new_hook_rot = [0.0, 0.0, -np.pi / 4]

        print("=> rotate:")
        hook_rot_pose = self.pos_rot_to_pose(
            (hook_pos[0], hook_pos[1] - 0.15, hook_pos[2] + 0.04), new_hook_rot)
        self.move_to(*hook_rot_pose, time_to_go=4.0)

        print("=> prehook:")
        prehook_pose = self.pos_rot_to_pose(
            (hook_pos[0] + 0.05, hook_pos[1] - 0.15, hook_pos[2] + 0.02), new_hook_rot)
        self.move_to(*prehook_pose)

        print("=> hook:")
        prehook_pose = self.pos_rot_to_pose(
            (hook_pos[0] + 0.02, hook_pos[1] - 0.15, hook_pos[2] + 0.02), new_hook_rot)
        self.move_to(*prehook_pose)

        print("=> open:")
        prehook_pose = self.pos_rot_to_pose(
            (hook_pos[0] + 0.02, hook_pos[1] - 0.07, hook_pos[2] + 0.07), new_hook_rot)
        self.move_to(*prehook_pose)

        print("=> back to rest pose:")
        self.move_to(self.rest_pos, self.rest_quat)

        self.reset()
        self.publish(3)


def main():
    # debug pick and place tools
    rospy.init_node('debug_manipulate', anonymous=True)

    robot = ManipulatorSystem()

    # rate = rospy.Rate(100)
    # while not rospy.is_shutdown():
    #     command = input("Please enter the name of the tool: \n")
    #     if command == 'c':
    #         break
    #     else:
    #         command_list = command.split('+')
    #         if command_list[0] == 'p':
    #             robot.put_back_tool(command_list[1])
    #         else:
    #             robot.take_away_tool(command_list[1], debug=False)

    #     rate.sleep()

    import random
    tool_list = ['gripper_asym', 'gripper_sym_plane', 'gripper_sym_rod', 
    'spatula_small', 'spatula_large', 
    'roller_large', 'press_circle', 'punch_circle',
    'roller_small', 'press_square', 'punch_square',
    'cutter_planar', 'cutter_circular', 'hook']
    random.shuffle(tool_list)

    tool_list = ['cutter_planar']

    for tool in tool_list:
        # robot.take_away_tool(tool, debug=False)
        robot.push([0.43, -0.1, 0.225], prepush_dh=0.07)
        # robot.put_back_tool(tool)


if __name__ == "__main__":
    main() 