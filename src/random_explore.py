# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import numpy as np
import os
import readchar
import rospy
import torch

from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
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

# Grasp params
n_grips = 3

grasp_force = 100.0
grasp_speed = 0.01

planner_dt = 0.02

ee_fingertip_T_mat = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
gripper_ft_trans = np.array([0.0, 0.019, 0.042192])

# 0 -> uninitialized / pause; 1 -> start; 2 -> stop
signal = 0

# Initialize interfaces
def main():
    global signal

    task_name = "cutting"
    tool = task_tool_mapping[task_name]
    rospy.init_node(task_name, anonymous=True)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        print(f"Run (r) or stop (c)?")
        key = readchar.readkey()
        print(key)
        if key == 'r':
            # take away the tool
            if robot.tool_status[tool] == 'ready':
                print(f"========== taking away {tool} ==========")
                robot.take_away_tool(tool)
                robot.tool_status[tool] = 'using'

            random_cut()
        elif key == 'c':
            if robot.tool_status[tool] == 'using':
                print(f"========== putting back {tool} ==========")
                robot.put_back_tool(tool)
                robot.tool_status[tool] = 'ready'
            
            break

        rate.sleep()


init_pose_seq = None
def init_pose_seq_callback(msg):
    global init_pose_seq
    init_pose_seq = msg.data.reshape(-1, 11, 14)


act_seq = None
def act_seq_callback(msg):
    global act_seq
    act_seq = msg.data.reshape(-1, 30, 12)


p_stats = None
def p_stats_callback(msg):
    global p_stats
    p_stats = msg.data
    print(f"p_stats: {p_stats}")


iter = 0
def iter_callback(msg):
    global iter
    iter = msg.data


def random_cut(pos_noise_scale=0.03):
    pos_noise_y = pos_noise_scale * (np.random.rand() * 2 - 1)
    print(pos_noise_y)
    cut_pos = [0.41, -0.1 + pos_noise_y, 0.23]
    cut_rot = [0.0, -0.05, np.pi / 4]
    precut_dh = 0.07
    robot.cut(cut_pos, cut_rot, precut_dh)


def random_grip():
    # Perform gripping
    p_mid_point = np.array([0.437, 0.0])
    p_noise_scale = 0.005
    plasticine_grasp_h = 0.1034 + 0.075 + 0.06 - 0.035, # ee_to_finger + finger_to_bot + cube_h + extra
    plasticine_pregrasp_dh = 0.1
    
    try:
        i = 0
        while True:
            # Sample grasp
            p_noise_x = p_noise_scale * (np.random.rand() * 2 - 1)
            p_noise_y = p_noise_scale * (np.random.rand() * 2 - 1)

            rot_noise = np.random.uniform(-np.pi, np.pi)
            
            if rot_noise > np.pi / 2:
                    rot_noise -= np.pi
            elif rot_noise < -np.pi / 2:
                rot_noise += np.pi

            grasp_params = (p_mid_point[0] + p_noise_x, p_mid_point[1] + p_noise_y, rot_noise)

            # Perform grasp
            print(f"========== Grasp {i+1} ==========")
            robot.grasp(grasp_params, plasticine_grasp_h, plasticine_pregrasp_dh)

            # Loop termination
            i += 1
            if n_grips > 0 and i >= n_grips:
                robot.reset()
                robot.signal_pub.publish(UInt8(2))
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")


def real_sim_remap(args, data, n_particle):
    points = data[0]
    # print(np.mean(points[:n_particle], axis=0), np.std(points[:n_particle], axis=0))
    points = (points - np.mean(points[:n_particle], axis=0)) / np.std(points[:n_particle], axis=0)
    points = np.array([points.T[0], points.T[2], points.T[1]]).T * args.std_p + args.mean_p

    n_shapes_floor = 9
    n_shapes_per_gripper = 11
    prim1 = points[n_particle + n_shapes_floor + 2] # + n_shapes_per_gripper // 2
    prim2 = points[n_particle + n_shapes_floor + n_shapes_per_gripper + 2]
    new_floor = np.array([[0.25, 0., 0.25], [0.25, 0., 0.5], [0.25, 0., 0.75],
                        [0.5, 0., 0.25], [0.5, 0., 0.5], [0.5, 0., 0.75],
                        [0.75, 0., 0.25], [0.75, 0., 0.5], [0.75, 0., 0.75]])
    new_prim1 = []
    for j in range(11):
        prim1_tmp = np.array([prim1[0], prim1[1] + 0.018 * (j - 5), prim1[2]])
        new_prim1.append(prim1_tmp)
    new_prim1 = np.stack(new_prim1)

    new_prim2 = []
    for j in range(11):
        prim2_tmp = np.array([prim2[0], prim2[1] + 0.018 * (j - 5), prim2[2]])
        new_prim2.append(prim2_tmp)

    new_prim2 = np.stack(new_prim2)
    new_state = np.concatenate([points[:n_particle], new_floor, new_prim1, new_prim2])

    return new_state


def sim_real_remap(init_pose_seq, act_seq):
    gripper_mid_point = int((init_pose_seq.shape[1] - 1) / 2)
    len_per_grip = 30
    sim_tool_size = 0.045
    robot_tool_size = 0.009
    sim_mean_p = np.array([0.5, 0.11348496, 0.5])
    sim_std_p = np.array([0.06, 0.04888084, 0.06])

    grasp_params_list = []
    grasp_width_list = []
    for i in range(init_pose_seq.shape[0]):
        rot_z = np.arctan2(init_pose_seq[i, gripper_mid_point, 9] - init_pose_seq[i, gripper_mid_point, 2], 
            init_pose_seq[i, gripper_mid_point, 7] - init_pose_seq[i, gripper_mid_point, 0]) + np.pi / 2 - np.pi / 4

        if rot_z > np.pi / 2:
            rot_z -= np.pi
        elif rot_z < -np.pi / 2:
            rot_z += np.pi

        gripper_grasp_pos_1 = init_pose_seq[i, gripper_mid_point, :3] + 0.02 * (np.sum(act_seq[i, :len_per_grip, :3], axis=0))
        if np.linalg.norm(act_seq[i, 0, :3]) > 0:
            gripper_grasp_pos_1 += sim_tool_size * act_seq[i, 0, :3] / np.linalg.norm(act_seq[i, 0, :3])
        gripper_grasp_pos_2 = init_pose_seq[i, gripper_mid_point, 7:10] + 0.02 * (np.sum(act_seq[i, :len_per_grip, 6:9], axis=0))
        if np.linalg.norm(act_seq[i, 0, 6:9]) > 0:
            gripper_grasp_pos_2 += sim_tool_size * act_seq[i, 0, 6:9] / np.linalg.norm(act_seq[i, 0, 6:9])

        gripper_init_pos_1 = (init_pose_seq[i, gripper_mid_point, :3] - sim_mean_p) / sim_std_p
        gripper_init_pos_1 = np.array([gripper_init_pos_1[0], gripper_init_pos_1[2], gripper_init_pos_1[1]]) * p_stats[3:] + p_stats[:3]
        gripper_init_pos_2 = (init_pose_seq[i, gripper_mid_point, 7:10] - sim_mean_p) / sim_std_p
        gripper_init_pos_2 = np.array([gripper_init_pos_2[0], gripper_init_pos_2[2], gripper_init_pos_2[1]]) * p_stats[3:] + p_stats[:3]

        gripper_grasp_pos_1 = (gripper_grasp_pos_1 - sim_mean_p) / sim_std_p
        gripper_grasp_pos_1 = np.array([gripper_grasp_pos_1[0], gripper_grasp_pos_1[2], gripper_grasp_pos_1[1]]) * p_stats[3:] + p_stats[:3]
        gripper_grasp_pos_2 = (gripper_grasp_pos_2 - sim_mean_p) / sim_std_p
        gripper_grasp_pos_2 = np.array([gripper_grasp_pos_2[0], gripper_grasp_pos_2[2], gripper_grasp_pos_2[1]]) * p_stats[3:] + p_stats[:3]

        mid_point = (gripper_init_pos_1 + gripper_init_pos_2) / 2

        grasp_width = np.linalg.norm(gripper_grasp_pos_1 - gripper_grasp_pos_2)
        grasp_width = grasp_width + 2 * robot_tool_size - 2 * gripper_ft_trans[1]
        grasp_width_list.append(grasp_width)

        grasp_params_list.append((mid_point[0], mid_point[1], rot_z))

    # x, y, rz
    return grasp_params_list, grasp_width_list


def replay(path):
    act_seq_path_list = sorted(glob.glob(os.path.join(path, 'act_seq_*')))
    init_pose_seq_path_list = sorted(glob.glob(os.path.join(path, 'init_pose_seq_*')))
    iter = 0
    init_pose_seq_replay = []
    act_seq_replay = []
    # import pdb; pdb.set_trace()
    for act_seq_path, init_pose_seq_path in zip(act_seq_path_list, init_pose_seq_path_list):
        init_pose_seq = np.load(init_pose_seq_path, allow_pickle=True)
        act_seq = np.load(act_seq_path, allow_pickle=True)
        # if iter == len(act_seq_path_list) - 1: 
        for i in range(init_pose_seq.shape[0]):
            init_pose_seq_replay.append(init_pose_seq[i])
            act_seq_replay.append(act_seq[i][:30])
        # else:
        #     init_pose_seq_replay.append(init_pose_seq[0])
        #     act_seq_replay.append(act_seq[0])

    return np.stack(init_pose_seq_replay), np.stack(act_seq_replay)


if __name__ == "__main__":
    main()
