import numpy as np

from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import *

# GNN planner:
# gripper_asym: dist_to_center, rot(0-2pi), grip_width
# gripper_sym_plane: dist_to_center, rot(0-pi), grip_width
# gripper_sym_rod: dist_to_center, rot(0-pi), grip_width
# press_circle / punch_circle: press_x, press_y, press_z
# press_square / punch_square: press_x, press_y, press_z, rot(0-pi/2)
# roller_small / roller_large: roll_x, roll_y, roll_z, rot(0-pi), roll_dist

# Precoded planner:
# pusher: push_x, push_y
# cutter_planar: cut_x, cut_y, cut_rot
# cutter_circular: cut_x, cut_y
# spatula: pick_x, pick_y, place_x, place_y
# hook: none

def grip(robot, params, grip_h=0.18, pregrip_dh=0.1):
    center_x, center_y, dist_to_center, rot, grip_width = params

    grip_pos_x = center_x - dist_to_center * np.sin(rot - np.pi / 4)
    grip_pos_y = center_y + dist_to_center * np.cos(rot - np.pi / 4)

    if rot > np.pi / 2:
        rot -= np.pi
    elif rot < -np.pi / 2:
        rot += np.pi
    
    robot.grip((grip_pos_x, grip_pos_y, rot), grip_h, pregrip_dh, grip_width)


def press(robot, params, prepress_dh=0.1):
    press_x, press_y, press_z, rot = params

    press_pos = [press_x, press_y, press_z]

    if rot > np.pi / 2:
        rot -= np.pi
    elif rot < -np.pi / 2:
        rot += np.pi

    press_rot = [0.0, 0.0, rot]

    robot.press(press_pos, press_rot, prepress_dh)


def roll(robot, params, preroll_dh=0.12):
    roll_x, roll_y, roll_z, rot, roll_dist = params
    start_pos = [roll_x, roll_y, roll_z]

    if rot > np.pi / 2:
        rot -= np.pi
    elif rot < -np.pi / 2:
        rot += np.pi

    roll_rot = [0.0, 0.0, rot]
    roll_delta = axangle2mat([0, 0, 1], roll_rot[2] + np.pi / 4) @ np.array([roll_dist, 0, 0]).T
    end_pos = start_pos + roll_delta

    robot.roll(start_pos, roll_rot, end_pos, preroll_dh)


def cut_planar(robot, params, cut_h=0.21, precut_dh=0.1, push_y=0.03):
    cut_x, cut_y, cut_rot = params
    robot.cut_planar([cut_x, cut_y, cut_h], [0.0, 0.0, cut_rot], precut_dh, push_y=push_y)


def cut_circular(robot, params, cut_h=0.18, precut_dh=0.1):
    cut_x, cut_y = params
    robot.cut_circular([cut_x, cut_y, cut_h], [0.0, 0.0, 0.0], precut_dh)


def push(robot, params, push_h=0.22, prepush_dh=0.1):
    push_x, push_y = params
    robot.push([push_x, push_y, push_h], prepush_dh)


def pick_and_place_skin(robot, params, grip_width, pick_h=0.18, prepick_dh=0.2, 
    place_h=0.2125, preplace_dh=0.1):
    pick_x, pick_y, place_x, place_y = params
    # robot.pick_and_place_skin_v1([pick_x, pick_y, pick_h], [0, 0, -np.pi / 4], prepick_dh, 
    #     [place_x, place_y, place_h], [0, 0, -np.pi / 4], preplace_dh, grip_width)
    robot.pick_and_place_skin_v2([pick_x, pick_y, pick_h], [0, 0, -np.pi / 4], prepick_dh, 
        [place_x, place_y, place_h], [0, 0, -np.pi / 4], preplace_dh, grip_width)


def pick_and_place_filling(robot, params, grip_width, pick_h=0.18, prepick_dh=0.2, 
    place_h=0.22, preplace_dh=0.05):
    pick_x, pick_y, place_x, place_y = params
    robot.pick_and_place_filling([pick_x, pick_y, pick_h], [0, 0, -np.pi / 4], prepick_dh, 
        [place_x, place_y, place_h], [0, 0, -np.pi / 4], preplace_dh, grip_width)


def hook(robot):
    robot.hook_dumpling_clip([0.4, -0.2, 0.2])
