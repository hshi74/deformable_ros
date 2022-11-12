import glob
import os
import random_explore
import readchar
import rosbag
import rospy
import sys
import tf2_ros
import yaml
import cv2

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import UInt8, Float32
from timeit import default_timer as timer
from transforms3d.quaternions import *


fixed_frame = 'panda_link0'
num_cams = 4

date = '11-01'

cd = os.path.dirname(os.path.realpath(sys.argv[0]))
tool_status_path = os.path.join(cd, '..', 'env', 'tool_status.yml')
data_path = os.path.join(cd, '..', 'raw_data', f'classifier_{date}')
# hdd_str = '/media/hxu/Game\ Drive\ PS4/robocook/raw_data'
# data_path = os.path.join(hdd_str, f'classifier_{date}')

os.system('mkdir -p ' + f"{data_path}")

# data_list = sorted(glob.glob(os.path.join(data_path.replace('\\', ''), '*')))
data_list = sorted(glob.glob(os.path.join(data_path, '*')))
if len(data_list) == 0:
    episode = 0
else:
    epi_prev = os.path.basename(data_list[-1]).split('_')[-1]
    episode = int(epi_prev[:-1].lstrip('0') + epi_prev[-1]) + 1


def get_tool_name():
    tool_name = ''
    
    with open(tool_status_path, 'r') as f:
        tool_status = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in tool_status.items():
        if value['status'] == 'using':
            tool_name = key
            break

    if len(tool_name) == 0:
        print('No tool is being used!')
        raise ValueError

    return tool_name


img_cls_signal = 0
pcd_cls_signal = 0
def action_signal_callback(msg):
    global img_cls_signal
    global pcd_cls_signal

    if msg.data == 0:
        img_cls_signal = 1
        pcd_cls_signal = 1

    if msg.data == 3:
        img_cls_signal = 2
        pcd_cls_signal = 2


episode_signal = 0
def episode_signal_callback(msg):
    global episode_signal
    # global episode
    # if episode_signal == 0 and msg.data == 1:
    #     episode += 1
    episode_signal = msg.data


img_done = 0
def image_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global img_cls_signal
    global img_done

    if img_cls_signal == 1 or img_cls_signal == 2:
        tool_name = get_tool_name()
        seq_path = os.path.join(data_path, f'ep_{str(episode).zfill(3)}', 
            f'seq_{str(seq).zfill(3)}_{tool_name}')
        os.system('mkdir -p ' + f"{seq_path}")

        image_msgs = [cam1_msg, cam2_msg, cam3_msg, cam4_msg]
        br = CvBridge()
        for i in range(num_cams):
            # Convert ROS Image message to OpenCV image
            img_bgr = br.imgmsg_to_cv2(image_msgs[i])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if img_cls_signal == 1:
                file_name = f'in_cam_{i+1}.png'
            else:
                file_name = f'out_cam_{i+1}.png'

            # cv2.imwrite(os.path.join(seq_path.replace('\\', ''), file_name), img_rgb)
            cv2.imwrite(os.path.join(seq_path, file_name), img_rgb)

        print(f"Ep {episode} Seq {seq}: recorded {'input' if img_cls_signal == 1 else 'output'} images")

        if img_cls_signal == 2:
            img_done = 1

        img_cls_signal = 0


pcd_done = 0
def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global pcd_cls_signal
    global pcd_done

    if pcd_cls_signal == 1 or pcd_cls_signal == 2:
        tool_name = get_tool_name()
        seq_path = os.path.join(data_path, f'ep_{str(episode).zfill(3)}', 
            f'seq_{str(seq).zfill(3)}_{tool_name}')
        os.system('mkdir -p ' + f"{seq_path}")

        if pcd_cls_signal == 1:
            file_name = f'in.bag'
        else:
            file_name = f'out.bag'

        # bag = rosbag.Bag(os.path.join(seq_path.replace('\\', ''), file_name), 'w')
        bag = rosbag.Bag(os.path.join(seq_path, file_name), 'w')
        bag.write('/cam1/depth/color/points', cam1_msg)
        bag.write('/cam2/depth/color/points', cam2_msg)
        bag.write('/cam3/depth/color/points', cam3_msg)
        bag.write('/cam4/depth/color/points', cam4_msg)

        bag.close()

        print(f"Ep {episode} Seq {seq}: recorded {'input' if pcd_cls_signal == 1 else 'output'} pcd")

        if pcd_cls_signal == 2:
            pcd_done = 1

        pcd_cls_signal = 0


seq = 0
def main():
    global cls_signal
    global episode
    global img_done
    global pcd_done
    global seq

    rospy.init_node("collect_cls_data", anonymous=True)
    rospy.Subscriber("/action_signal", UInt8, action_signal_callback)
    rospy.Subscriber("/episode_signal", UInt8, episode_signal_callback)

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
        if img_done == 1 and pcd_done == 1:
            episode_len = 5 if 'gripper' in get_tool_name() else 3
            episode += (seq + 1) // episode_len
            seq = (seq + 1) % episode_len
            img_done = 0
            pcd_done = 0

        rate.sleep()


if __name__ == '__main__':
    main()
