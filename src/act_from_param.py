import numpy as np
import fcl
import yaml

from transforms3d.euler import euler2quat
from transforms3d.quaternions import *

# gripper_sym_rod: 
#   r: the radial distance
#   theta: azimuthal angle
#   phi: polar angle
#   grip_width: minimum distance during the grip

ee_fingertip_T_mat = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
# 5cm: 0.042192, 7cm: 0.052192, 8cm: 0.057192
tool_center_z = 0.057192
# 5cm: 0.0656, 7cm: 0.0856, 8cm: 0.0956
finger_obj_list = [
    fcl.CollisionObject(fcl.Cylinder(0.009, 0.0956), fcl.Transform()),
    fcl.CollisionObject(fcl.Cylinder(0.009, 0.0956), fcl.Transform())
]

finger_center_T_list = [
    np.concatenate((np.concatenate((np.eye(3), np.array([[0.0, -0.019, tool_center_z]]).T), axis=1), [[0, 0, 0, 1]]), axis=0),
    np.concatenate((np.concatenate((np.eye(3), np.array([[0.0, 0.019, tool_center_z]]).T), axis=1), [[0, 0, 0, 1]]), axis=0)
]

support = fcl.Cylinder(0.06, 0.08)
support_T = fcl.Transform(np.array([0.435, -0.01, 0.04]))
support_obj = fcl.CollisionObject(support, support_T)

dough_bbox = fcl.Box(0.1, 0.1, 0.08)
dough_bbox_T = fcl.Transform(np.array([0.435, -0.01, 0.12]))
dough_bbox_obj = fcl.CollisionObject(dough_bbox, dough_bbox_T)


def get_fingermid_pos(center, r, theta, phi):
    center_x, center_y, center_z = center
    pos_x = center_x + (r + tool_center_z) * np.sin(phi) * np.cos(theta - np.pi / 2)
    pos_y = center_y + (r + tool_center_z) * np.sin(phi) * np.sin(theta - np.pi / 2)
    pos_z = center_z + (r + tool_center_z) * np.cos(phi) 
    return np.array([pos_x, pos_y, pos_z])


def set_finger_transform(grip_width, fingertip_mat, fingermid_pos):
    global finger_obj_list

    for k in range(len(finger_obj_list)):
        fingertip_pos = (fingertip_mat @ np.array([0, (2 * k - 1) * (grip_width / 2), 0]).T).T + fingermid_pos
        fingertip_T = np.concatenate((np.concatenate((fingertip_mat, np.array([fingertip_pos]).T), axis=1), 
            [[0, 0, 0, 1]]), axis=0) @ finger_center_T_list[k]
        finger_T = fcl.Transform(fingertip_T[:3, :3], fingertip_T[:3, 3].T)
        finger_obj_list[k].setTransform(finger_T)


def grip(robot, center, params, threshold=0.007):
    succeeded = 1

    r, theta, phi, grip_width = params

    robot.rotate_stand(theta)

    fingermid_pos = get_fingermid_pos(center, r, 0, phi)
    ee_quat = qmult([0, 1, 0, 0], qmult(axangle2quat([0, 0, 1], 0 - np.pi / 4), 
        axangle2quat([1, 1, 0], phi)))
    ee_pos = fingermid_pos - (quat2mat(ee_quat) @ ee_fingertip_T_mat[:3, 3].T).T 
    # import pdb; pdb.set_trace()
    print(f'center: {center}\nparams: {params}\nfingermid:{fingermid_pos}\nee_pos: {ee_pos}\nee_quat: {ee_quat}')

    fingertip_mat = quat2mat(ee_quat) @ ee_fingertip_T_mat[:3, :3]

    set_finger_transform(grip_width, fingertip_mat, fingermid_pos)

    ret_min = float('inf')
    for k in range(len(finger_obj_list)):
        ret = fcl.distance(support_obj, finger_obj_list[k], fcl.DistanceRequest(), fcl.DistanceResult())
        ret_min = min(ret, ret_min)

    print(f"min distance: {ret_min}")
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

        fingermid_pos = get_fingermid_pos(center, r + dist, 0, phi)
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
