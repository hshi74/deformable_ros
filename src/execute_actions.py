# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import numpy as np
import os
import rospy
import torch

from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from timeit import default_timer as timer
from transforms3d.quaternions import *

from manipulate import ManipulatorSystem

# Grasp params
n_grips = 3

p_mid_point = np.array([0.437, 0.0])
p_noise_scale = 0.005
rest_pose = ([p_mid_point[0], 0.0, 0.6], [1.0, 0.0, 0.0, 0.0])
plasticine_grasp_h = 0.1034 + 0.075 + 0.06 # ee_to_finger + finger_to_bot + cube_h + extra
plasticine_pregrasp_dh = 0.1
grasp_force = 100.0
grasp_speed = 0.01

planner_dt = 0.02

ee_fingertip_T_mat = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
gripper_ft_trans = np.array([0.0, 0.019, 0.042192])


def random_pose(rot_prev, i):
    p_noise_x = p_noise_scale * (np.random.rand() * 2 - 1)
    p_noise_y = p_noise_scale * (np.random.rand() * 2 - 1)

    rot_noise = np.random.uniform(-np.pi, np.pi)
    
    if rot_noise > np.pi / 2:
            rot_noise -= np.pi
    elif rot_noise < -np.pi / 2:
        rot_noise += np.pi

    # if i < n_grips - 1:
    #     rot_noise = np.pi / 2
    # else:
    #     rot_noise = np.random.uniform(-np.pi, 0)
    # rot_curr = rot_prev + rot_noise

    # x, y, rz
    return p_mid_point[0] + p_noise_x, p_mid_point[1] + p_noise_y, rot_noise


def grasp_random():
    # Perform grasping
    i = 0
    rot_curr = np.random.uniform(-np.pi / 2, 0)
    try:
        while True:
            # Sample grasp
            grasp_params = random_pose(rot_curr, i)
            rot_curr = grasp_params[-1]

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


def grasp_as_plan(mode):
    global init_pose_seq
    global act_seq
    global p_stats

    global iter
    
    # prefix = "/scr/hxu/projects/deformable/VGPL-Dynamics-Prior/dump/dump_ngrip_fixed_robot_v4/" + \
    #         "files_dy_21-Jan-2022-22:33:11.729243_nHis4_aug0.05_gt0_seqlen6_emd0.9_chamfer0.1_uh0.0_clip0.0/"
    # control_out_dir = prefix + "control_robot/X/selected"
    control_out_dir = "/scr/hxu/projects/deformable/VGPL-Dynamics-Prior/dump/robot_control_rebuttal/A"
    
    print(f"Mode: {mode}")
    if mode == "react":
        pass
    elif mode == 'read':
        init_pose_seq = np.load(f"{control_out_dir}/init_pose_seq_opt.npy", allow_pickle=True)
        act_seq = np.load(f"{control_out_dir}/act_seq_opt.npy", allow_pickle=True)
    elif mode == 'replay':
        p_stats = np.load(f"{control_out_dir}/p_stats.npy", allow_pickle=True)
        init_pose_seq, act_seq = replay(control_out_dir)
    else:
        raise NotImplementedError

    rospy.Subscriber("/init_pose_seq", numpy_msg(Floats), init_pose_seq_callback)
    rospy.Subscriber("/act_seq", numpy_msg(Floats), act_seq_callback)
    rospy.Subscriber("/p_stats", numpy_msg(Floats), p_stats_callback)
    rospy.Subscriber("/iter", UInt8, iter_callback)

    # Perform grasping
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        # print("Spinning...")
        if not (init_pose_seq is None or act_seq is None):
            print(init_pose_seq.shape, act_seq.shape)
            grasp_params_list, grasp_width_list = sim_real_remap(init_pose_seq, act_seq)
            print(grasp_params_list, grasp_width_list)

            if mode == "replay":
                robot.signal_pub.publish(UInt8(1))

            for i in range(len(grasp_params_list)):
                # Perform grasp
                # last = i == len(grasp_params_list) - 1
                print(grasp_params_list[i], plasticine_grasp_h, plasticine_pregrasp_dh, grasp_width_list[i])
                robot.grasp(grasp_params_list[i], plasticine_grasp_h, plasticine_pregrasp_dh, 
                            grasp_width_list[i], mode=mode)

            robot.reset()

            init_pose_seq = None
            act_seq = None

            if mode == "replay":
                robot.signal_pub.publish(UInt8(0))
            
            if mode == "react":
                robot.signal_pub.publish(UInt8(1))

        rate.sleep()


# Initialize interfaces
robot = ManipulatorSystem()
def main():
    global robot

    rospy.init_node('grasp', anonymous=True)

    # grasp_random()
    grasp_as_plan('react')


if __name__ == "__main__":
    main()
