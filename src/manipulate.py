import copy
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
from pyquaternion import Quaternion
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from timeit import default_timer as timer
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import *


cd = os.path.dirname(os.path.realpath(sys.argv[0]))


class ManipulatorSystem:
    def __init__(self):
        self.arm = RobotInterface(
            ip_address="192.168.0.2",
            enforce_version=False,
        )

        self.gripper = GripperInterface(
            ip_address="192.168.0.2",
        )

        # 0: start, 1: end, 2: away
        self.action_signal_pub = rospy.Publisher('/action_signal', UInt8, queue_size=10)

        # Reset to rest pose
        self.rest_pose = self.pos_rz_to_pose((0.45, 0.0, np.pi / 4), 0.45)
        self.rest_pos = self.rest_pose[0]
        self.rest_quat = self.rest_pose[1]

        self.grip_speed = 0.05
        self.grip_force = 20.0
        self.max_width = 0.08

        self.planner_dt = 0.02
        self.gain_multiplier = 4


    def __del__(self):
        self.arm.terminate_current_policy()


    def reset_policy(self):
        self.arm.start_cartesian_impedance(
            self.gain_multiplier * torch.Tensor(self.arm.metadata.default_Kx), 
            self.gain_multiplier * torch.Tensor(self.arm.metadata.default_Kxd)
        )


    def publish(self, signal):
        self.action_signal_pub.publish(UInt8(signal))
        time.sleep(0.2)


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


    def move_to(self, pos, quat, time_to_go=2.0):
        # Plan trajectory
        N = int(time_to_go / self.planner_dt)

        pos_curr, quat_curr = self.arm.get_ee_pose()
        waypoints = toco.planning.generate_cartesian_space_min_jerk(
            start=T.from_rot_xyz(R.from_quat(quat_curr), pos_curr),
            goal=T.from_rot_xyz(R.from_quat(quat), pos),
            time_to_go=time_to_go,
            hz=1 / self.planner_dt,
        )
       
        # Execute trajectory
        t0 = time.time()
        t_target = t0
        successes = 0
        error_detected = False
        robot_states = []
        for i in range(N):
            # Update traj
            try:
                ee_pos_desired = waypoints[i]["pose"].translation()
                ee_quat_desired = waypoints[i]["pose"].rotation().as_quat()

                self.arm.update_desired_ee_pose(position=ee_pos_desired, orientation=ee_quat_desired)
            except Exception as e:
                error_detected = True
                print(f"Error updating current policy {str(e)}")

            # Check if policy terminated due to issues
            if self.arm.get_previous_interval().end != -1 or error_detected:
                error_detected = False
                print("Interrupt detected. Reinstantiating control policy...")
                self.reset_policy()
                time.sleep(1)
                break
            else:
                successes += 1

            # Spin once
            t_target += self.planner_dt
            t_remaining = t_target - time.time()
            time.sleep(max(t_remaining, 0.0))

        # Wait for robot to stabilize
        time.sleep(0.2)

        pos_actual, quat_actual = self.arm.get_ee_pose()
        # print(f'Actual pose: {pos_actual}, {quat_actual}')
        pos_err = round(np.linalg.norm(pos_actual - pos), 3)
        quat_err = round(Quaternion.absolute_distance(Quaternion(array=quat_actual), 
            Quaternion(array=quat)), 3)
        print(f'\tpose error: {pos_err}, {quat_err}')

        return successes, N, robot_states


    def close_gripper(self, grip_width, blocking=True, grip_params=None):
        if grip_params:
            self.gripper.goto(width=grip_width, speed=grip_params[0], force=grip_params[1], blocking=blocking)
        else:
            self.gripper.goto(width=grip_width, speed=self.grip_speed, force=self.grip_force, blocking=blocking)

        time.sleep(0.2)

        # Check state
        state = self.gripper.get_state()
        assert state.width < self.max_width


    def open_gripper(self, blocking=True, grip_params=None):
         # self.gripper.get_state().max_width
        if grip_params:
            self.gripper.goto(width=self.max_width, speed=grip_params[0], force=grip_params[1], blocking=blocking)
        else:
            self.gripper.goto(width=self.max_width, speed=self.grip_speed, force=self.grip_force, blocking=blocking)

        time.sleep(0.2)

        # Check state
        state = self.gripper.get_state()
        assert state.width > 0.0


    def pose_to_pose(self, grip_pose):
        pos = torch.Tensor(grip_pose[0])
        quat = torch.Tensor([*grip_pose[1][1:], grip_pose[1][0]])

        return pos, quat


    def pos_rz_to_pose(self, grip_pose, z):
        x, y, rz = grip_pose
        pos = torch.Tensor([x, y, z])
        quat = (
            R.from_rotvec(torch.Tensor([0, 0, rz])) * R.from_quat(torch.Tensor([1, 0, 0, 0]))
        ).as_quat()

        return pos, quat


    # quaternion is w, x, y, z 
    def grip(self, grip_pose, grip_width, pregrip_dh=0.1):
        self.reset_policy()

        grip_pose = self.pose_to_pose(grip_pose)
        # rotate_pose = (self.rest_pos, copy.deepcopy(grip_pose[1]))
        idle_pose = copy.deepcopy(grip_pose)
        idle_pose[0][2] += pregrip_dh

        # print("=> rotate:")
        self.open_gripper()
        # self.move_to(*rotate_pose, time_to_go=4.0)
        
        print("=> get ready:")
        self.move_to(*idle_pose, time_to_go=4.0)

        print("=> grip:")
        self.move_to(*grip_pose, time_to_go=2.0)

        self.publish(1)

        # Grip
        self.close_gripper(grip_width, blocking=True, grip_params=(0.02, 150))
        time.sleep(6.0)

        self.publish(0)

        # Release
        self.open_gripper(blocking=True, grip_params=(0.05, 150))
        time.sleep(4.0)

        print("=> back to idle pose:")
        self.move_to(*idle_pose, time_to_go=4.0)

        # print("=> back to rotated pose:")
        # self.move_to(*rotate_pose, time_to_go=2.0)

        print("=> back to rest pose:")
        self.move_to(*self.rest_pose, time_to_go=4.0)


    # grip_params is x, y, rz 
    def rotate_stand(self, theta_delta, center_xy=(0.43, -0.008), grip_h=0.28, grip_width=0.07, pregrip_dh=0.07):
        self.reset_policy()

        # this number should be in the range of (-pi, pi]
        with open(os.path.join(cd, '..', 'env/stand_theta.txt'), 'r') as f:
            theta_cur = float(f.read())

        theta_robot = theta_cur + np.pi / 4
        theta_robot_new = theta_cur + theta_delta + np.pi / 4 

        if theta_robot > np.pi / 2:
            theta_robot -= np.pi
        elif theta_robot < -np.pi / 2:
            theta_robot += np.pi
        
        if theta_robot_new > np.pi / 2:
            theta_robot_new -= np.pi
        elif theta_robot_new < -np.pi / 2:
            theta_robot_new += np.pi

        grip_pose = self.pos_rz_to_pose((*center_xy, theta_robot), grip_h)
        idle_pose = copy.deepcopy(grip_pose)
        idle_pose[0][2] += pregrip_dh

        rotate_pose = self.pos_rz_to_pose((*center_xy, theta_robot_new), grip_h)
        release_pose = copy.deepcopy(rotate_pose)
        release_pose[0][2] += pregrip_dh

        self.publish(2)

        # print("=> rotate:")
        self.open_gripper()
        # self.move_to(*rotate_pose, time_to_go=4.0)

        self.publish(0)

        print("=> get ready:")
        self.move_to(*idle_pose, time_to_go=4.0)

        print("=> grip:")
        self.move_to(*grip_pose, time_to_go=2.0)

        # Grip
        self.close_gripper(grip_width, blocking=True)
        time.sleep(2.0)

        print("=> rotate:")
        self.move_to(*rotate_pose, time_to_go=4.0)

        # Release
        self.open_gripper(blocking=True)
        time.sleep(2.0)

        print("=> back to release pose:")
        self.move_to(*release_pose, time_to_go=2.0)

        print("=> back to rest pose:")
        self.move_to(*self.rest_pose, time_to_go=4.0)

        theta_new = (theta_cur + theta_delta) % (2 * np.pi)
        if theta_new > np.pi:
            theta_new -= np.pi

        with open(os.path.join(cd, '..', 'env/stand_theta.txt'), 'w') as f:
            f.write(str(theta_new))


def main():
    # debug pick and place tools
    rospy.init_node('debug_manipulate', anonymous=True)

    robot = ManipulatorSystem()


if __name__ == "__main__":
    main() 