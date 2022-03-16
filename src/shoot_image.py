from pickletools import uint8
import time
import numpy as np
import os
import readchar
import rosbag
import rospy
import ros_numpy
import sys
import tf
import tf2_ros
import yaml
import cv2

from cv_bridge import CvBridge
from datetime import datetime
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from timeit import default_timer as timer
from transforms3d.quaternions import *

tool_list = ['planar_cutter', 'asym_gripper', 'roller', 'stamp']
# init_shape_list = ['cube', 'rope', 'ball', 'plane', 'random']

tool_idx = 3
# init_shape_idx = 0

fixed_frame = 'panda_link0'
num_cams = 4

cd = os.path.dirname(os.path.realpath(sys.argv[0]))
prefix = os.path.join('tool_classification_pre_3-15', tool_list[tool_idx])
image_path = os.path.join(cd, '..', 'dataset', prefix)

os.system('mkdir -p ' + f"{image_path}")

trial = 0
signal = 0
input = True

def image_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global signal
    global trial

    if signal == 1:
        # bag = rosbag.Bag(os.path.join(bag_path, f'{init_shape_list[init_shape_idx]}_{trial}.bag'), 'w')
        # bag.write('/cam1/color/image_raw', cam1_msg)
        # bag.write('/cam2/color/image_raw', cam2_msg)
        # bag.write('/cam3/color/image_raw', cam3_msg)
        # bag.write('/cam4/color/image_raw', cam4_msg)
        # bag.close()

        image_msgs = [cam1_msg, cam2_msg, cam3_msg, cam4_msg]
        br = CvBridge()
        for i in range(num_cams):
            # Convert ROS Image message to OpenCV image
            img_bgr = br.imgmsg_to_cv2(image_msgs[i])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            file_name = str(trial).zfill(3)
            # folder_path = os.path.join(image_path, str(trial).zfill(3))
            if input:
                file_name += '_in'
            else:
                file_name += '_out'
            # os.system('mkdir -p ' + f"{folder_path}")
            cv2.imwrite(os.path.join(image_path, file_name + f'_cam_{i+1}.png'), img_rgb)

        if not input: 
            trial += 1

        signal = 0


def main():
    global signal
    global input

    rospy.init_node('image_shooter', anonymous=True)

    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/color/image_raw", Image), 
        Subscriber("/cam2/color/image_raw", Image), 
        Subscriber("/cam3/color/image_raw", Image), 
        Subscriber("/cam4/color/image_raw", Image)),
        queue_size=10,
        slop=0.1
    )

    tss.registerCallback(image_callback)

    with open(os.path.join(os.path.join(cd, '..', 'env'), 'camera_pose_world.yml'), 'r') as f:
        camera_pose_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    static_br = tf2_ros.StaticTransformBroadcaster()
    static_ts_list = []
    for i in range(1, num_cams + 1):
        static_ts = TransformStamped()
        # static_ts.header.stamp = rospy.Time.now()
        static_ts.header.frame_id = fixed_frame
        static_ts.child_frame_id = f"cam{i}_link"

        static_ts.transform.translation.x = camera_pose_dict[f"cam_{i}"]["position"][0]
        static_ts.transform.translation.y = camera_pose_dict[f"cam_{i}"]["position"][1]
        static_ts.transform.translation.z = camera_pose_dict[f"cam_{i}"]["position"][2]

        static_ts.transform.rotation.x = camera_pose_dict[f"cam_{i}"]["orientation"][1]
        static_ts.transform.rotation.y = camera_pose_dict[f"cam_{i}"]["orientation"][2]
        static_ts.transform.rotation.z = camera_pose_dict[f"cam_{i}"]["orientation"][3]
        static_ts.transform.rotation.w = camera_pose_dict[f"cam_{i}"]["orientation"][0]

    static_br.sendTransform(static_ts_list)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        key = readchar.readkey()
        print(f"Trial {trial}: Input or output? ")
        print(key)
        if key == 'i' or key == 'o':
            signal = 1
            input = False
            if key == 'i':
                print("Input!")
                input = True
            else:
                print("Output!")
        elif key == 'c':
            break

        rate.sleep()


if __name__ == '__main__':
    main()
