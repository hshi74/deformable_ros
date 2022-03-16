from pickletools import uint8
import time
import numpy as np
import os
import readchar
import rosbag
import rospy
import sys
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from timeit import default_timer as timer
from transforms3d.quaternions import *

tool_list = ['planar_cutter', 'circular_cutter', 'asym_gripper', 'roller', 'stamp']
init_shape_list = ['cube', 'rope', 'ball', 'plane', 'random']

tool_idx = 0
init_shape_idx = 0

fixed_frame = 'panda_link0'
num_cams = 4

cd = os.path.dirname(os.path.realpath(sys.argv[0]))
prefix = os.path.join('tool_classification_pre_3-15', tool_list[tool_idx])
bag_path = os.path.join(cd, '..', 'dataset', prefix, f'cube_0.bag')

def main():
    bag = rosbag.Bag(bag_path)

    image_msgs = []
    for topic, msg, t in bag.read_messages(
        topics=['/cam1/color/image_raw', '/cam2/color/image_raw', '/cam3/color/image_raw', '/cam4/color/image_raw']
    ):
        image_msgs.append(msg)

    bag.close()

    br = CvBridge()
    for i in range(len(image_msgs)):
        # Convert ROS Image message to OpenCV image
        img_bgr = br.imgmsg_to_cv2(image_msgs[i])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(cd, '..', 'dataset', prefix, f'image_{i}.png'), img_rgb)


if __name__ == '__main__':
    main()
