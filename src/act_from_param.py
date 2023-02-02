import numpy as np
import fcl
import rospy
import yaml

from transforms3d.euler import euler2quat
from transforms3d.quaternions import *

from utils import get_cube_center

# gripper_sym_rod: 
#   r: the radial distance
#   theta: azimuthal angle
#   phi: polar angle
#   grip_width: minimum distance during the grip

ee_fingertip_T_mat = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
# 5cm: 0.042192, 7cm: 0.052192, 8cm: 0.057192
tool_center_z = 0.06472
# 5cm: 0.0656, 7cm: 0.0856, 8cm: 0.0956
finger_obj_list = [
    fcl.CollisionObject(fcl.Cylinder(0.009, 0.0806), fcl.Transform()),
    fcl.CollisionObject(fcl.Cylinder(0.009, 0.0806), fcl.Transform())
]

finger_center_T_list = [
    np.concatenate((np.concatenate((np.eye(3), np.array([[0.0, -0.019, tool_center_z]]).T), axis=1), [[0, 0, 0, 1]]), axis=0),
    np.concatenate((np.concatenate((np.eye(3), np.array([[0.0, 0.019, tool_center_z]]).T), axis=1), [[0, 0, 0, 1]]), axis=0)
]

support = fcl.Cylinder(0.06, 0.08)
support_T = fcl.Transform(np.array([0.43, -0.01, 0.04]))
support_obj = fcl.CollisionObject(support, support_T)

dough_bbox = fcl.Box(0.1, 0.1, 0.08)
dough_bbox_T = fcl.Transform(np.array([0.43, -0.01, 0.12]))
dough_bbox_obj = fcl.CollisionObject(dough_bbox, dough_bbox_T)


cube = None
def get_cube(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global cube

    pcd_msgs = [cam1_msg, cam2_msg, cam3_msg, cam4_msg]
    cube, _ = get_cube_center(pcd_msgs, visualize=False)


def wait_for_visual():
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        if cube is not None:
            break

        rate.sleep()


def get_fingermid_pos(center, r, dist, phi1, phi2, theta=0):
    center_x, center_y, center_z = center
    pos_x = center_x + (-r * np.sin(phi1) + dist * np.sin(phi2)) * np.cos(theta - np.pi / 2) 
    pos_y = center_y + (-r * np.sin(phi1) + dist * np.sin(phi2)) * np.sin(theta - np.pi / 2)
    pos_z = center_z + r * np.cos(phi1) + dist * np.cos(phi2)
    
    # pos_x = center_x + (r * np.sin(phi1) + dist * 0.0) * np.sin(theta)
    # pos_y = center_y + (r * np.sin(phi1) + dist * 0.0) * np.cos(theta)
    # pos_z = center_z + max(r, 0) * np.cos(phi1) + dist * 1.0 + 0.027
    
    return np.array([pos_x, pos_y, pos_z])


def set_finger_transform(grip_width, fingertip_mat, fingermid_pos):
    global finger_obj_list

    for k in range(len(finger_obj_list)):
        fingertip_pos = (fingertip_mat @ np.array([0, (2 * k - 1) * (grip_width / 2), 0]).T).T + fingermid_pos
        # fingertip_pos = (fingertip_mat @ np.array([(1 - 2 * k) * (grip_width / 2), 0, 0]).T).T + fingermid_pos
        fingertip_T = np.concatenate((np.concatenate((fingertip_mat, np.array([fingertip_pos]).T), axis=1), 
            [[0, 0, 0, 1]]), axis=0) @ finger_center_T_list[k]
        finger_T = fcl.Transform(fingertip_T[:3, :3], fingertip_T[:3, 3].T)
        finger_obj_list[k].setTransform(finger_T)


center = None
def grip(robot, params, threshold=0.007):
    global center

    succeeded = 1

    r, theta, phi1, phi2, grip_width = params

    grip_width = max(0.001, grip_width - 0.015)

    robot.rotate_stand(theta)
    theta = 0

    wait_for_visual()
    center = cube.get_center()

    fingermid_pos = get_fingermid_pos(center, r, tool_center_z, phi1, phi2, theta)
    ee_quat = qmult([0, 1, 0, 0], qmult(axangle2quat([0, 0, 1], theta - np.pi / 4), 
        axangle2quat([1, 1, 0], phi2)))
    # ee_quat = qmult([0, 1, 0, 0], axangle2quat([0, 0, 1], theta - np.pi / 4))
    ee_pos = fingermid_pos - (quat2mat(ee_quat) @ ee_fingertip_T_mat[:3, 3].T).T 
    # import pdb; pdb.set_trace()
    print(f'center: {center}\nparams: {params}\nfingermid:{fingermid_pos}\nee_pos: {ee_pos}\nee_quat: {ee_quat}')

    fingertip_mat = quat2mat(ee_quat) @ ee_fingertip_T_mat[:3, :3]

    # fingertip_mat = ee_fingertip_T_mat[:3, :3] @ quat2mat(ee_quat)

    set_finger_transform(grip_width, fingertip_mat, fingermid_pos)

    ret_min = float('inf')
    for k in range(len(finger_obj_list)):
        ret = fcl.distance(support_obj, finger_obj_list[k], fcl.DistanceRequest(), fcl.DistanceResult())
        ret_min = min(ret, ret_min)

    # print(f"min distance: {ret_min}")
    iter = 0
    dist = 0.0
    while ret_min < 0.5 * threshold or (iter > 0 and ret_min > threshold):
        if ret_min < threshold:
            if ret_min < 0:
                dist += 2 * threshold
            else:
                dist += 0.2 * threshold
        else:
            dist -= 0.2 * threshold

        fingermid_pos = get_fingermid_pos(center, r, tool_center_z + dist, phi1, phi2)
        set_finger_transform(grip_width, fingertip_mat, fingermid_pos)

        ret_min = float('inf')
        for k in range(len(finger_obj_list)):
            ret = fcl.distance(support_obj, finger_obj_list[k], fcl.DistanceRequest(), fcl.DistanceResult())
            ret_min = min(ret, ret_min)

        if iter > 20:
            print("Unable to figure out a good distance!")
            succeeded = 0
            break

        # import pdb; pdb.set_trace()
        print(f'min distance: {ret_min}\nadjusted distance: {dist}')

    # print(f'min distance: {ret_min}\nadjusted distance: {dist}')

    for k in range(len(finger_obj_list)):
        ret_dough = fcl.distance(dough_bbox_obj, finger_obj_list[k], fcl.DistanceRequest(), fcl.DistanceResult())
        if ret_dough > 0.0:
            print("Too far from the dough!")
            succeeded = 0
            break

    if succeeded:
        ee_pos = fingermid_pos - (quat2mat(ee_quat) @ ee_fingertip_T_mat[:3, 3].T).T 
        robot.grip((ee_pos, ee_quat), grip_width)

    return succeeded, fingermid_pos
