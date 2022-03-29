import time
import numpy as np
import os

import rosbag
import rospy
import ros_numpy
import sys
import tf
import tf2_ros
import yaml

from datetime import datetime
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from timeit import default_timer as timer
from transforms3d.quaternions import *

import manipulate

robot = manipulate.ManipulatorSystem()

fixed_frame = 'panda_link0'
task_name = 'cutting_pre_3-26'
num_cams = 4
cd = os.path.dirname(os.path.realpath(sys.argv[0]))
data_path = os.path.join(cd, '..', 'dataset', task_name)
os.system('mkdir -p ' + f"{data_path}")
with open(os.path.join(os.path.join(cd, '..', 'env'), 'camera_pose_world.yml'), 'r') as f:
    cam_pose_dict = yaml.load(f, Loader=yaml.FullLoader)


# 0 -> uninitialized / pause; 1 -> start; 2 -> stop
signal = 0
def signal_callback(msg):
    global signal
    signal = msg.data


time_start = 0.0
time_last = 0.0
time_now = 0.0
trial = 0

def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global signal
    global time_start
    global time_last
    global time_now
    global trial

    if signal == 1:
        if time_start == 0.0:
            time_start = cam1_msg.header.stamp.to_sec()

        time_now = cam1_msg.header.stamp.to_sec() - time_start
        # print(time_now - time_last)
        if time_now == 0.0 or time_now - time_last > 0.1:
            trial_path = os.path.join(data_path, str(trial).zfill(3))
            os.system('mkdir -p ' + f"{trial_path}")
            bag = rosbag.Bag(os.path.join(trial_path, f'{time_now:.3f}.bag'), 'w')
            bag.write('/cam1/depth/color/points', cam1_msg)
            bag.write('/cam2/depth/color/points', cam2_msg)
            bag.write('/cam3/depth/color/points', cam3_msg)
            bag.write('/cam4/depth/color/points', cam4_msg)

            ee_pose = robot.get_ee_pose()
            bag.write('/ee_pose', ee_pose)

            bag.close()

            print(f"Trial {trial}: recorded pcd at {time_now}...")
            time_last = time_now

    if signal == 0 and time_start > 0:
        time_start = 0.0
        time_last = 0.0
        time_now = 0.0
        trial += 1


def main():
    global signal
    rospy.init_node('point_cloud_writer', anonymous=True)

    rospy.Subscriber("/signal", UInt8, signal_callback)

    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/depth/color/points", PointCloud2), 
        Subscriber("/cam2/depth/color/points", PointCloud2), 
        Subscriber("/cam3/depth/color/points", PointCloud2), 
        Subscriber("/cam4/depth/color/points", PointCloud2)),
        queue_size=10,
        slop=0.1
    )

    tss.registerCallback(cloud_callback)

    static_br = tf2_ros.StaticTransformBroadcaster()
    static_ts_list = []
    for i in range(1, num_cams + 1):
        static_ts = TransformStamped()
        # static_ts.header.stamp = rospy.Time.now()
        static_ts.header.frame_id = fixed_frame
        static_ts.child_frame_id = f"cam{i}_link"

        static_ts.transform.translation.x = cam_pose_dict[f"cam_{i}"]["position"][0]
        static_ts.transform.translation.y = cam_pose_dict[f"cam_{i}"]["position"][1]
        static_ts.transform.translation.z = cam_pose_dict[f"cam_{i}"]["position"][2]

        static_ts.transform.rotation.x = cam_pose_dict[f"cam_{i}"]["orientation"][1]
        static_ts.transform.rotation.y = cam_pose_dict[f"cam_{i}"]["orientation"][2]
        static_ts.transform.rotation.z = cam_pose_dict[f"cam_{i}"]["orientation"][3]
        static_ts.transform.rotation.w = cam_pose_dict[f"cam_{i}"]["orientation"][0]
        
        static_ts_list.append(static_ts)

    static_br.sendTransform(static_ts_list)

    # br = tf.TransformBroadcaster()
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        if signal == 2: break
        rate.sleep()


if __name__ == '__main__':
    main()
