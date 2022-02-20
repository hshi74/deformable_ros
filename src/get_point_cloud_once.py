from pickletools import uint8
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

import grasp

fixed_frame = 'panda_link0'
task_name = 'ngrip_fixed_robot_1-25'
num_cams = 4
cd = os.path.dirname(os.path.realpath(sys.argv[0]))
time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
bag_path = os.path.join(cd, '..', 'dataset', task_name, time_now)
os.system('mkdir -p ' + f"{bag_path}")
with open(os.path.join(os.path.join(cd, '..', 'env'), 'cam_pose_new.yml'), 'r') as f:
    cam_pose_dict = yaml.load(f, Loader=yaml.FullLoader)


# 0 -> uninitialized / pause; 1 -> start; 2 -> stop
signal = 1
iter = 0
def signal_callback(msg):
    global signal
    global iter
    signal = msg.data
    iter += 1


def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global signal
    global iter

    if signal == 1:
        bag = rosbag.Bag(os.path.join(bag_path, f'plasticine_{iter}.bag'), 'w')
        bag.write('/cam1/depth/color/points', cam1_msg)
        bag.write('/cam2/depth/color/points', cam2_msg)
        bag.write('/cam3/depth/color/points', cam3_msg)
        bag.write('/cam4/depth/color/points', cam4_msg)

        gripper_1_pose, gripper_2_pose = grasp.robot.get_gripper_pose()
        bag.write('/gripper_1_pose', gripper_1_pose)
        bag.write('/gripper_2_pose', gripper_2_pose)

        bag.close()

        signal = 0


def main():
    global signal
    rospy.init_node('point_cloud_writer', anonymous=True)

    iter_pub = rospy.Publisher('/iter', UInt8, queue_size=10)
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

        static_ts.transform.translation.x = cam_pose_dict[f"cam_{i}"][0]
        static_ts.transform.translation.y = cam_pose_dict[f"cam_{i}"][1]
        static_ts.transform.translation.z = cam_pose_dict[f"cam_{i}"][2]

        static_ts.transform.rotation.x = cam_pose_dict[f"cam_{i}"][3]
        static_ts.transform.rotation.y = cam_pose_dict[f"cam_{i}"][4]
        static_ts.transform.rotation.z = cam_pose_dict[f"cam_{i}"][5]
        static_ts.transform.rotation.w = cam_pose_dict[f"cam_{i}"][6]
        
        static_ts_list.append(static_ts)

    static_br.sendTransform(static_ts_list)

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        iter_pub.publish(UInt8(iter))
        rate.sleep()


if __name__ == '__main__':
    main()
