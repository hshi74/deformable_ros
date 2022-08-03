import glob
import os
import readchar
import rosbag
import rospy
import sys
import tf2_ros
import yaml
import cv2

from cv_bridge import CvBridge
from datetime import datetime
from manipulate import ManipulatorSystem
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import UInt8, Float32
from timeit import default_timer as timer
from transforms3d.quaternions import *


# tool_list = ['gripper_asym', 'gripper_sym_plane', 'gripper_sym_rod', 'gripper_sym_spatula', 
#     'roller_large', 'stamp_circle_large', 'stamp_circle_small',
#     'roller_small', 'stamp_square_large', 'stamp_square_small',
#     'cutter_planar', 'cutter_circular', 'hook']

tool_name = 'gripper_sym'

fixed_frame = 'panda_link0'
num_cams = 4

cd = os.path.dirname(os.path.realpath(sys.argv[0]))
image_suffix = os.path.join('tool_classifier_raw_image_08-02', tool_name)
pcd_suffix = os.path.join('tool_classifier_raw_pcd_08-02', tool_name)
image_path = os.path.join(cd, '..', 'raw_data', image_suffix)
pcd_path = os.path.join(cd, '..', 'raw_data', pcd_suffix)

os.system('mkdir -p ' + f"{image_path}")
os.system('mkdir -p ' + f"{pcd_path}")

robot = ManipulatorSystem()

data_list = sorted(glob.glob(os.path.join(pcd_path, '*')))
if len(data_list) == 0:
    trial = 0
else:
    trial = int(os.path.basename(data_list[-1]).lstrip('0').split('_')[0]) + 1

img_cls_signal = 0
pcd_cls_signal = 0

img_done = 0
pcd_done = 0


def cls_signal_callback(msg):
    global img_cls_signal
    global pcd_cls_signal
    img_cls_signal = msg.data
    pcd_cls_signal = msg.data


def image_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global img_cls_signal
    global img_done

    if img_cls_signal == 1 or img_cls_signal == 2:
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

            if img_cls_signal == 1:
                file_name += '_in'
            else:
                file_name += '_out'

            cv2.imwrite(os.path.join(image_path, file_name + f'_cam_{i+1}.png'), img_rgb)

        print(f"Trial {trial}: recorded {'input' if img_cls_signal == 1 else 'output'} images")

        if img_cls_signal == 2:
            img_done = 1

        img_cls_signal = 0


def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global pcd_cls_signal
    global pcd_done

    if pcd_cls_signal == 1 or pcd_cls_signal == 2:
        file_name = str(trial).zfill(3)
        if pcd_cls_signal == 1:
            file_name += '_in'
        else:
            file_name += '_out'

        data_path = os.path.join(pcd_path, f'{file_name}.bag')
        bag = rosbag.Bag(data_path, 'w')
        bag.write('/cam1/depth/color/points', cam1_msg)
        bag.write('/cam2/depth/color/points', cam2_msg)
        bag.write('/cam3/depth/color/points', cam3_msg)
        bag.write('/cam4/depth/color/points', cam4_msg)

        ee_pose = robot.get_ee_pose()
        bag.write('/ee_pose', ee_pose)

        gripper_width = robot.gripper.get_state().width
        bag.write('/gripper_width', Float32(gripper_width))

        bag.close()

        print(f"Trial {trial}: recorded {'input' if pcd_cls_signal == 1 else 'output'} pcd")

        if pcd_cls_signal == 2:
            pcd_done = 1

        pcd_cls_signal = 0


def main():
    global cls_signal
    global img_done
    global pcd_done
    global trial

    rospy.init_node('image_shooter', anonymous=True)
    rospy.Subscriber("/cls_signal", UInt8, cls_signal_callback)
    
    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/color/image_raw", Image), 
        Subscriber("/cam2/color/image_raw", Image), 
        Subscriber("/cam3/color/image_raw", Image), 
        Subscriber("/cam4/color/image_raw", Image)),
        queue_size=10,
        slop=0.1
    )

    tss.registerCallback(image_callback)

    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/depth/color/points", PointCloud2), 
        Subscriber("/cam2/depth/color/points", PointCloud2), 
        Subscriber("/cam3/depth/color/points", PointCloud2), 
        Subscriber("/cam4/depth/color/points", PointCloud2)),
        queue_size=100,
        slop=0.2
    )

    tss.registerCallback(cloud_callback)

    with open(os.path.join(cd, '..', 'env', 'camera_pose_world.yml'), 'r') as f:
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
        # print(f"Trial {trial}: Input or output? ")
        # key = readchar.readkey()
        # print(key)
        # if key == 'i' or key == 'o':
        #     signal = 1
        #     input = False
        #     if key == 'i':
        #         print("Input!")
        #         input = True
        #     else:
        #         print("Output!")
        # elif key == 'c':
        #     break

        if img_done == 1 and pcd_done == 1:
            trial += 1
            img_done = 0
            pcd_done = 0

        rate.sleep()


if __name__ == '__main__':
    main()
