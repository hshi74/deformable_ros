import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import readchar
import rospy
import ros_numpy
import sys
import yaml

from act_from_param import *
from datetime import datetime
from manipulate import ManipulatorSystem
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from transforms3d.quaternions import *
from utils import get_cube_center


robot = ManipulatorSystem()

hdd_str = '/media/hshi74/wd_drive/robocraft3d/raw_data'
data_path = os.path.join(hdd_str, f'gripper_sym_rod_robot_v1')

os.system('mkdir -p ' + f"{data_path}")

data_list = sorted(glob.glob(os.path.join(data_path, '*')))
if len(data_list) > 0:
    epi_prev = os.path.basename(data_list[-1]).split('_')[-1]
    episode = int(epi_prev[:-1].lstrip('0') + epi_prev[-1]) + 1
else:
    episode = 0

seq = 0

pcd_signal = 0
cube = None
def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global pcd_signal
    global cube

    if pcd_signal == 1:
        pcd_msgs = [cam1_msg, cam2_msg, cam3_msg, cam4_msg]
        cube, _ = get_cube_center(pcd_msgs, visualize=False)

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


center = None
def random_explore():
    global pcd_signal
    global center
    global episode

    episode_signal_pub = rospy.Publisher('/episode_signal', UInt8, queue_size=10)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        print(f"run (r) or stop (c)?")
        key = readchar.readkey()
        print(key)
        if key == 'r':
            episode_signal_pub.publish(UInt8(0))
            random_grip(5)
            episode += 1
            episode_signal_pub.publish(UInt8(1))
        elif key == 'c':
            break
        else:
            print('Unrecoganized command!')

        rate.sleep()


def random_grip(n_grips):
    global seq
    r_range = (0, 0.03)
    theta_range = (-np.pi / 2, np.pi / 2)
    phi_range = (-np.pi / 2, np.pi / 2) # with respect to +z
    grip_width_range = (0.004, 0.02)

    # r_list = (0.01, 0.01, 0.01, 0.01, 0.01)
    # theta_list = (0.0, 0.0, 0.0, -np.pi / 2, np.pi / 2)
    # phi_list = (0.0, -np.pi / 2, np.pi / 2, 0.0, 0.0)

    # succ_grip_list = []
    # fail_grip_list = []
    seq = 0
    for i in range(n_grips):
        wait_for_visual()
        # center = [0.45, 0.0, 0.1] 
        center = get_center(bbox=False)
        # print(f"The center of the play_doh is {center}")

        # r = r_list[i]
        # theta = theta_list[i]
        # phi = phi_list[i]
        r = np.random.uniform(*r_range)
        theta = np.random.uniform(*theta_range)
        phi = np.random.uniform(*phi_range)
        grip_width = np.random.uniform(*grip_width_range)

        params = (r, theta, phi, grip_width)

        seq_path = os.path.join(data_path, f'ep_{str(episode).zfill(3)}', f'seq_{str(seq).zfill(3)}')
        os.system('mkdir -p ' + f"{seq_path}")
        
        with open(os.path.join(seq_path, 'param_seq.npy'), 'wb') as f:
            np.save(f, np.array(params))

        print(f"===== Grip {i+1}: {params} =====")

        ret = grip(robot, center, params)

        seq += 1

        # if ret[0]:
        #     succ_grip_list.append((params, ret[1]))
        # else:
        #     fail_grip_list.append((params, ret[1]))

    # visualize(succ_grip_list, fail_grip_list, path=f'debug/grips_{datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")}.png')


# def visualize(succ_grip_list, fail_grip_list, path=''):
#     fig = plt.figure(figsize=(20, 10))
#     ax_params = fig.add_subplot(121, projection='3d')
#     ax_params.set_xlabel('r')
#     ax_params.set_ylabel('theta')
#     ax_params.set_zlabel('phi')

#     ax_pos = fig.add_subplot(122, projection='3d')
#     ax_pos.set_xlabel('x')
#     ax_pos.set_ylabel('y')
#     ax_pos.set_zlabel('z')

#     if len(succ_grip_list) > 0:
#         succ_params_list, succ_ret_list = zip(*succ_grip_list)
#         ax_params.scatter(*zip(*succ_params_list), c='b', s=30, label='success')
#         ax_pos.scatter(*zip(*succ_ret_list), c='b', s=30)
    
#     if len(fail_grip_list) > 0:
#         fail_params_list, fail_ret_list = zip(*fail_grip_list)
#         ax_params.scatter(*zip(*fail_params_list), c='r', s=30, label='failure')
#         ax_pos.scatter(*zip(*fail_ret_list), c='r', s=30)

#     ax_params.legend()

#     if len(path) > 0:
#         plt.savefig(path)
#     else:
#         plt.show()

#     plt.close()


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

    random_explore()


if __name__ == "__main__":
    main()
