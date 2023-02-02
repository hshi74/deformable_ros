import glob
import os
import readchar
import rosbag
import rospy
import sys
import tf2_ros
import yaml

from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import UInt8, Float32
from timeit import default_timer as timer
from transforms3d.quaternions import *


fixed_frame = 'panda_link0'
num_cams = 4

cd = os.path.dirname(os.path.realpath(sys.argv[0]))

target_name = 'pagoda'
data_path = os.path.join(f'/scr/hshi74/projects/robocraft3d/target_shapes/3d_real/{target_name}')

os.system('mkdir -p ' + f"{data_path}")

pcd_target_signal = 0
def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global pcd_target_signal

    if pcd_target_signal == 1 or pcd_target_signal == 2:
        if pcd_target_signal == 1:
            file_name = f'000.bag'
        else:
            file_name = f'001.bag'

        bag = rosbag.Bag(os.path.join(data_path, file_name), 'w')
        bag.write('/cam1/depth/color/points', cam1_msg)
        bag.write('/cam2/depth/color/points', cam2_msg)
        bag.write('/cam3/depth/color/points', cam3_msg)
        bag.write('/cam4/depth/color/points', cam4_msg)

        bag.close()

        pcd_target_signal = 0


def main():
    global pcd_target_signal

    rospy.init_node("collect_cls_data", anonymous=True)

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
        print(f"initial (1) or target (2) or end (3)?")
        key = readchar.readkey()
        print(key)
        pcd_target_signal = int(key)
        if pcd_target_signal == 3:
            break

        print(f"Recorded {'initial' if pcd_target_signal == 1 else 'target'} pcd")

        rate.sleep()


if __name__ == '__main__':
    main()
