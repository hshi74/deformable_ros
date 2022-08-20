import numpy as np
import os
import readchar
import rospy
import open3d as o3d
import ros_numpy
import sys
import yaml

from act_from_param import *
from manipulate import ManipulatorSystem
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import *
from utils import get_cube


robot = ManipulatorSystem()

pcd_signal = 0
cube = None
def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global pcd_signal
    global cube
    global center

    if pcd_signal == 1:
        pcd_msgs = [cam1_msg, cam2_msg, cam3_msg, cam4_msg]
        cube = get_cube(pcd_msgs, target_color='white')

        pcd_signal = 0


def wait_for_visual():
    global pcd_signal

    pcd_signal = 1
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        if pcd_signal == 0:
            break

        rate.sleep()


def get_center(bbox=True):
    if bbox:
        bbox = cube.get_axis_aligned_bounding_box()
        return bbox.get_center()
    else:
        return cube.get_center()


def get_cut_params(target_volume=5e-5, visualize=False):
    bbox = cube.get_axis_aligned_bounding_box()
    
    if visualize:
        o3d.visualization.draw_geometries([cube, bbox])
    
    volume_cur = bbox.volume()
    x_len, y_len, z_len = bbox.get_extent()
    x_center, y_center, z_center = bbox.get_center()

    target_y_len = target_volume / volume_cur * y_len

    target_x = x_center
    target_y = y_center - y_len / 2 + target_y_len
    
    return [target_x, target_y, np.pi / 4]


def make_dumpling(debug=True):
    global pcd_signal

    # for key, value in robot.tool_status.items():
    #     if value['status'] == 'using':
    #         print(f"===== putting back {key} =====")
    #         robot.put_back_tool(key)
    #         break

    # print(f"===== Step 1: Cut the dough to the right volumes =====")
    # robot.take_away_tool('cutter_planar')
    # wait_for_visual()
    # params = get_cut_params()
    # print(f"===== Cut params: {params} =====")
    # cut_planar(robot, params, push_y=0.2)
    
    # robot.put_back_tool('cutter_planar')

    # print(f"===== Step 2: grip the dough to a cube =====")
    # robot.take_away_tool('gripper_sym_plane')
    # wait_for_visual()
    # center = get_center()
    # param_seq = [[center[0], center[1], 0, np.pi/4, 0.01], [center[0], center[1], 0, -np.pi/4, 0.01]]
    # for params in param_seq:
    #     grip(robot, params)

    # robot.put_back_tool('gripper_sym_plane')

    # print(f"===== Step 3: press the dough to a flat cube =====")
    # robot.take_away_tool('press_square')
    # wait_for_visual()
    # center = get_center()
    # param_seq = [[center[0], center[1], 0.069 + 0.1034 + 0.01, 0]]
    # for params in param_seq:
    #     press(robot, params)

    # robot.put_back_tool('press_square')

    # print(f"===== Step 4: roll the dough to a flat skin =====")
    # robot.take_away_tool('roller_large')
    # wait_for_visual()
    # center = get_center(bbox=False)
    # param_seq = [
    #     [center[0], center[1], 0.089 + 0.1034 + 0.015, -np.pi / 4, -0.04],
    #     [center[0], center[1], 0.089 + 0.1034 + 0.015, np.pi / 4, -0.04],
    #     [center[0], center[1], 0.089 + 0.1034 + 0.015, -np.pi / 4, 0.04],
    #     [center[0], center[1], 0.089 + 0.1034 + 0.015, np.pi / 4, 0.04]
    # ]
    # for params in param_seq:
    #     roll(robot, params)

    # robot.put_back_tool('roller_large')

    print(f"===== Step 5: cut the dumpling skin =====")
    robot.take_away_tool('cutter_circular')
    wait_for_visual()
    # this center will be reused for the following actions
    center = get_center(bbox=False)
    cut_circular(robot, center[:2])
    
    robot.put_back_tool('cutter_circular')

    print(f"===== Step 6: push away undesired dough =====")
    robot.take_away_tool('cutter_planar')
    # wait_for_visual()
    # center = get_center(bbox=False)
    push(robot, center[:2])
    robot.put_back_tool('cutter_planar')

    print(f"===== Step 7: pick and place the dumping skin =====")
    robot.take_away_tool('spatula_small')
    # wait_for_visual()
    # center = get_center(bbox=False)
    pick_and_place_skin(robot, [*center[:2], 0.395, -0.29], 0.005)
    robot.put_back_tool('spatula_small')

    print(f"===== Step 7: pick and place the dumping filling =====")
    robot.take_away_tool('spatula_large')
    # wait_for_visual()
    pick_and_place_filling(robot, [0.5, -0.3, 0.395, -0.29], 0.015)
    robot.put_back_tool('spatula_large')

    print(f"===== Step 8: hook and close the dumpling clip =====")
    robot.take_away_tool('hook')
    hook(robot)
    robot.put_back_tool('hook')


def main():
    rospy.init_node('random_explore', anonymous=True)

    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/depth/color/points", PointCloud2), 
        Subscriber("/cam2/depth/color/points", PointCloud2), 
        Subscriber("/cam3/depth/color/points", PointCloud2), 
        Subscriber("/cam4/depth/color/points", PointCloud2)),
        queue_size=100,
        slop=0.2
    )

    tss.registerCallback(cloud_callback)

    make_dumpling()


if __name__ == "__main__":
    main()
