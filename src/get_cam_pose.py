#!/usr/bin/env python

import math
import numpy as np
import os
import roma
import rospy
import sys
import tf
import torch
import yaml
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import Pose
from tqdm import tqdm
from transforms3d.quaternions import *


# set the robot base as (0,0,0)
# def update_ar_tag_pose(id, args):
#     robot_state_dict, ar_tag_dist_dict, ar_tag_ori = args
#     br = tf.TransformBroadcaster()
#     ar_tag_pos_world = (quat2mat(ar_tag_ori) @ np.array(ar_tag_dist_dict[id][:3]).T).T + np.array(robot_state_dict['fingertip_pos'])
#     ar_tag_ori_world = qmult(ar_tag_ori, np.array(ar_tag_dist_dict[id][3:]))
#     # import pdb; pdb.set_trace()
#     br.sendTransform(tuple(ar_tag_pos_world), tuple(ar_tag_ori_world), rospy.Time.now(), f"ar_marker_{id}", "panda_link0")

def rotate(quat, pos):
    return torch.t(torch.mm(roma.unitquat_to_rotmat(quat), torch.t(pos.unsqueeze(0)))).squeeze()


def get_tag_pose(pose):
    pos = np.array([pose.position.x, pose.position.y, pose.position.z])
    # WXYZ convention
    ori = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
    # cam_pos = quat2mat(qinverse(ori)) @ -pos
    # cam_ori = qinverse(ori)
    return np.concatenate((pos, ori))


tag_pose_dict = {}
def ar_tag_callback(data):
    global tag_pose_dict
    for marker in data.markers:
        # rospy.loginfo(rospy.get_caller_id() + f"I heard {marker.id} has {marker.pose}")
        tag_pose = get_tag_pose(marker.pose.pose)
        if marker.id in tag_pose_dict:
            tag_pose_dict[marker.id].append(tag_pose)
        else:
            tag_pose_dict[marker.id] = [tag_pose]
        # update_ar_tag_pose(marker.id, args)


def main():
    global tag_pose_dict
    rospy.init_node('ar_pose_subscriber', anonymous=True)

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_dir = os.path.join(cd, '..', 'env')
    with open(os.path.join(data_dir, 'robot_state.yml'), 'r') as f:
        robot_state_dict = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(data_dir, 'ar_tag_dist.yml'), 'r') as f:
        ar_tag_dist_dict = yaml.load(f, Loader=yaml.FullLoader)

    # fixed_frame = 'panda_link0'
    # ar tag orientation in the fixed frame
    ar_tag_ori = roma.rotvec_to_unitquat(torch.tensor([0, 0, math.pi / 2]))

    rospy.Subscriber("/ar_pose_marker", AlvarMarkers, ar_tag_callback)

    pose_num = 50
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        done = False
        for tag_id_init, pose_list in tag_pose_dict.items():
            done = len(pose_list) >= pose_num
            break
        
        if done:
            tag_pos_mean_dict = {}
            tag_ori_mean_dict = {}
            for tag_id, pose_list in tag_pose_dict.items():
                if not tag_id in range(18):
                    continue
                pose_list = np.array(pose_list)
                tag_pos_mean = np.mean(pose_list[:, :3], axis=0)
                # How to average quaternions: https://stackoverflow.com/a/49690919
                # q(ndarray): an Mx4 ndarray of quaternions.
                # w(list): an M elements list, a weight for each quaternion.
                q = pose_list[:, 3:]
                w = [1 / q.shape[0] for _ in range(q.shape[0])]
                tag_ori_mean = np.linalg.eigh(np.einsum('ij,ik,i->...jk', q, q, w))[1][:, -1]
                # import pdb; pdb.set_trace()
                tag_pos_mean_dict[tag_id] = torch.tensor(tag_pos_mean, dtype=torch.float32)
                # Quaternion -> XYZW convention
                tag_ori_mean_dict[tag_id] = torch.tensor(
                    [tag_ori_mean[1], tag_ori_mean[2], tag_ori_mean[3], tag_ori_mean[0]], dtype=torch.float32
                )
            # print(f"Pos: {cam_pos_mean_dict}\nOri: {cam_ori_mean_dict}")
            break

        rate.sleep()

    # print(ar_tag_dist_dict)
    # import pdb; pdb.set_trace()
    tag_id_init = list(tag_pos_mean_dict.keys())[0]
    ar_tag_dist_init = torch.tensor(ar_tag_dist_dict[tag_id_init])
    cam_pos_init = rotate(roma.quat_inverse(tag_ori_mean_dict[tag_id_init]), -tag_pos_mean_dict[tag_id_init])
    cam_pos_init = rotate(ar_tag_dist_init[3:], cam_pos_init) + ar_tag_dist_init[:3]
    cam_ori_init = roma.quat_inverse(tag_ori_mean_dict[tag_id_init])
    cam_ori_init = roma.quat_product(ar_tag_dist_init[3:], cam_ori_init)

    cam_pos = cam_pos_init.requires_grad_()
    cam_ori = cam_ori_init.requires_grad_()
    
    # optimizer = torch.optim.SGD([cam_pos, cam_ori], lr=0.0001)

    # n_epoch = 1000
    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1500, gamma=0.1)
    # for _ in tqdm(range(n_epoch), total=n_epoch):
    #     optimizer.zero_grad()
    #     loss = 0.0
    #     cam_ori_norm = cam_ori / torch.linalg.norm(cam_ori)
    #     # import pdb; pdb.set_trace()
    #     ar_tag_pos_init = rotate(roma.quat_inverse(cam_ori_norm), -cam_pos)
    #     ar_tag_ori_init = roma.quat_inverse(cam_ori_norm)
    #     for tag_id in tag_pos_mean_dict.keys():
    #         # import pdb; pdb.set_trace()
    #         ar_tag_dist = torch.tensor(ar_tag_dist_dict[tag_id])
    #         ar_tag_ori_predicted = roma.quat_product(ar_tag_dist[3:], ar_tag_ori_init)
    #         ar_tag_pos_predicted = rotate(ar_tag_ori_predicted, ar_tag_dist[:3]) + ar_tag_pos_init

    #         pos_d = torch.linalg.norm(tag_pos_mean_dict[tag_id] - ar_tag_pos_predicted)
    #         ori_d = roma.quat_inverse(tag_ori_mean_dict[tag_id]) * ar_tag_ori_predicted
    #         ori_d = 2 * torch.atan2(torch.linalg.norm(ori_d[:3]), ori_d[3]) * np.pi / 180
    #         print(f"pos diff: {pos_d}; ori diff: {ori_d}")
    #         loss += pos_d + ori_d
    #     print(f"loss: {loss}")
    #     loss.backward()
    #     optimizer.step()
    #     # scheduler.step()

    # print(f"Pos: {cam_pos_mean_dict}\nOri: {cam_ori_mean_dict}")
    # cam_pos_mean = np.mean(list(cam_pos_mean_dict.values()), axis=0)
    # q = np.array(list(cam_ori_mean_dict.values()))
    # w = [1 / q.shape[0] for _ in range(q.shape[0])]
    # cam_ori_mean = np.linalg.eigh(np.einsum('ij,ik,i->...jk', q, q, w))[1][:, -1]
    # # print(f"Pos mean: {cam_pos_mean}\nOri mean: {cam_ori_mean}")
    
    cam_idx = rospy.get_param('camera_idx')
    # br = tf.TransformBroadcaster()
    # while not rospy.is_shutdown():
    # import pdb; pdb.set_trace()
    cam_pos_world = rotate(ar_tag_ori, cam_pos) + torch.tensor(robot_state_dict['fingertip_pos'])
    cam_ori_world = roma.quat_product(ar_tag_ori, cam_ori)
    cam_ori_world = cam_ori_world / torch.linalg.norm(cam_ori_world)
    # br.sendTransform(tuple(cam_pos_world), (cam_ori_world[1], cam_ori_world[2], cam_ori_world[3], cam_ori_world[0]), rospy.Time.now(), f"cam{cam_idx}_link", fixed_frame)
    # rate.sleep()

    cam_pos_world = cam_pos_world.cpu().detach().numpy()
    cam_ori_world = cam_ori_world.cpu().detach().numpy()

    print(f"Pos: {cam_pos_world}\nOri: {cam_ori_world}")
    flag = 'a' if os.path.exists(os.path.join(data_dir, 'cam_pose.yml')) else 'w'
    with open(os.path.join(data_dir, 'cam_pose.yml'), flag) as f:
        cam_pose_world = np.concatenate((cam_pos_world, cam_ori_world))
        yaml.dump({f'cam_{cam_idx}': cam_pose_world.tolist()}, f)

if __name__ == '__main__':
    main()
