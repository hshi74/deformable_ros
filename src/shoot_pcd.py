import execute_actions
import manipulate
import numpy as np
import os
import random_explore
import rosbag
import rospy
import sys
import timeit
import tf2_ros
import yaml

from datetime import datetime
from message_filters import ApproximateTimeSynchronizer, Subscriber
from timeit import default_timer as timer
from transforms3d.quaternions import *

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8, Float32
from robocook_ros.msg import RobotPose


if len(sys.argv) < 2:
    print("Please enter the mode!")
    exit()

mode = sys.argv[1] # collect_data, record_result, or correct_control

fixed_frame = 'panda_link0'
num_cams = 4

# task_name = 'cutting_pre_3-26'
# task_name = 'gripping_pre_4-21'
# task_name = 'rolling_pre_4-29'
task_name = 'pressing_pre_4-29'

cd = os.path.dirname(os.path.realpath(sys.argv[0]))
data_path = os.path.join(cd, '..', 'raw_data', task_name)
os.system('mkdir -p ' + f"{data_path}")

# 0 -> uninitialized / pause; 1 -> start; 2 -> stop
if mode == 'collect_data':
    robot = random_explore.robot
    signal = 0
elif mode == 'record_result':
    robot = manipulate.ManipulatorSystem()
    datetime_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    os.system('mkdir -p ' + f"{os.path.join(data_path, datetime_now)}")
    signal = 0
elif mode == 'correct_control':
    robot = execute_actions.robot
    datetime_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    os.system('mkdir -p ' + f"{os.path.join(data_path, datetime_now)}")
    signal = 1
else:
    raise NotImplementedError

with open(os.path.join(os.path.join(cd, '..', 'env'), 'camera_pose_world.yml'), 'r') as f:
    cam_pose_dict = yaml.load(f, Loader=yaml.FullLoader)


def main():
    global signal
    rospy.init_node('point_cloud_writer', anonymous=True)

    iter_pub = rospy.Publisher('/iter', UInt8, queue_size=10)
    # robot_pose_pub = rospy.Publisher('/robot_pose', RobotPose, queue_size=10)
    rospy.Subscriber("/signal", UInt8, signal_callback)
    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/depth/color/points", PointCloud2), 
        Subscriber("/cam2/depth/color/points", PointCloud2), 
        Subscriber("/cam3/depth/color/points", PointCloud2), 
        Subscriber("/cam4/depth/color/points", PointCloud2)),
        # Subscriber("/robot_pose", RobotPose)),
        queue_size=100,
        slop=0.2
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
        # t0 = timer()
        # robot_pose_msg = RobotPose()
        # robot_pose_msg.header.stamp = rospy.Time.now()
        # robot_pose_msg.ee_pose = robot.get_ee_pose()
        # robot_pose_msg.gripper_width = Float32(robot.gripper.get_state().width)

        # robot_pose_pub.publish(robot_pose_msg)
        # t1 = timer()
        # print(f'Time taken to publish: {t1 - t0}')
        
        if mode == 'correct_control':
            iter_pub.publish(UInt8(iter))

        if signal == 2: break
        
        rate.sleep()


iter = 0
def signal_callback(msg):
    global signal
    global iter
    signal = msg.data
    iter += 1


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
    global iter

    if mode == 'collect_data' or mode == 'record_result' :
        if signal == 1:
            if time_start == 0.0:
                time_start = cam1_msg.header.stamp.to_sec()

            time_now = cam1_msg.header.stamp.to_sec() - time_start
            time_diff_1 = cam2_msg.header.stamp.to_sec() - time_start - time_now
            time_diff_2 = cam3_msg.header.stamp.to_sec() - time_start - time_now
            time_diff_3 = cam4_msg.header.stamp.to_sec() - time_start - time_now
            # time_diff_4 = robot_pose_msg.header.stamp.to_sec() - time_start - time_now
            # print(time_now - time_last)
            if time_now == 0.0 or time_now - time_last > 0.1:
                if mode == 'collect_data':
                    trial_path = os.path.join(data_path, str(trial).zfill(3))
                    os.system('mkdir -p ' + f"{trial_path}")
                    bag = rosbag.Bag(os.path.join(trial_path, f'{time_now:.3f}.bag'), 'w')
                    print(f"Trial {trial}: recorded pcd at {round(time_now, 3)} " + \
                        f"({round(time_diff_1, 3)}, {round(time_diff_2, 3)}, {round(time_diff_3, 3)})...")
                else:
                    bag = rosbag.Bag(os.path.join(data_path, datetime_now, f'{time_now:.3f}.bag'), 'w')
                
                # t0 = timer()
                bag.write('/cam1/depth/color/points', cam1_msg)
                bag.write('/cam2/depth/color/points', cam2_msg)
                bag.write('/cam3/depth/color/points', cam3_msg)
                bag.write('/cam4/depth/color/points', cam4_msg)

                # bag.write('/robot_pose', robot_pose_msg)
                # t1 = timer()
                ee_pose = robot.get_ee_pose()
                bag.write('/ee_pose', ee_pose)

                gripper_width = robot.gripper.get_state().width
                bag.write('/gripper_width', Float32(gripper_width))

                # print(f'Time taken to write: {t1 - t0}')

                bag.close()

                time_last = time_now

        if mode == 'collect_data' and signal == 0 and time_start > 0:
            trial += 1
            time_start = 0.0
            time_last = 0.0
            time_now = 0.0

    elif mode == 'correct_control':
        if signal == 1:
            bag = rosbag.Bag(os.path.join(data_path, datetime_now, f'state_{iter}.bag'), 'w')
            bag.write('/cam1/depth/color/points', cam1_msg)
            bag.write('/cam2/depth/color/points', cam2_msg)
            bag.write('/cam3/depth/color/points', cam3_msg)
            bag.write('/cam4/depth/color/points', cam4_msg)

            # bag.write('/robot_pose', robot_pose_msg)
            ee_pose = robot.get_ee_pose()
            bag.write('/ee_pose', ee_pose)

            gripper_width = robot.gripper.get_state().width
            bag.write('/gripper_width', Float32(gripper_width))

            bag.close()

            signal = 0

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
