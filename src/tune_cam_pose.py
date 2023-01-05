import copy
import time
import numpy as np
import os
import open3d as o3d
import readchar
import rosbag
import rospy
import ros_numpy
import sys
import tf
import tf2_ros
import yaml

from datetime import datetime
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from timeit import default_timer as timer
from transforms3d.quaternions import *
from visualization_msgs.msg import Marker, MarkerArray


fixed_frame = 'panda_link0'
num_cams = 4
cd = os.path.dirname(os.path.realpath(sys.argv[0]))
data_dir = os.path.join(cd, '..', 'env')
with open(os.path.join(data_dir, 'camera_pose_world.yml'), 'r') as f:
    camera_pose_dict = yaml.load(f, Loader=yaml.FullLoader)

camera_pose_dict_new = copy.deepcopy(camera_pose_dict)

pos_stride = 0.0005
rot_stride = 0.001

tune_idx = [3]
marker_array_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=100)

signal = 0

def cloud_callback(cam1_msg, cam2_msg, cam3_msg, cam4_msg):
    global signal
    if signal == 1:
        markers = get_plane_markers([cam1_msg, cam2_msg, cam3_msg, cam4_msg])
        # print(markers)
        for i, m in enumerate(markers):
            if i + 1 in tune_idx:
                marker_array_pub.publish(m)

        signal = 0


def main():
    global signal

    rospy.init_node('cam_pose_tuner', anonymous=True)

    tss = ApproximateTimeSynchronizer(
        (Subscriber("/cam1/depth/color/points", PointCloud2), 
        Subscriber("/cam2/depth/color/points", PointCloud2), 
        Subscriber("/cam3/depth/color/points", PointCloud2), 
        Subscriber("/cam4/depth/color/points", PointCloud2)),
        queue_size=100,
        slop=0.2
    )

    tss.registerCallback(cloud_callback)

    static_br = tf2_ros.StaticTransformBroadcaster()
    static_ts_list = []
    for i in [1, 2, 3, 4]:
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
        
        static_ts_list.append(static_ts)

    static_br.sendTransform(static_ts_list)

    pcd_trans_vec = [0.0, 0.0, 0.0]
    pcd_rot_vec = [0.0, 0.0, 0.0]

    br = tf.TransformBroadcaster()
    rate = rospy.Rate(100)
    save = False
    while not rospy.is_shutdown():
        key = readchar.readkey()
        if key == 'w':
            pcd_trans_vec[0] += pos_stride
        elif key == 'x':
            pcd_trans_vec[0] -= pos_stride
        elif key == 'a':
            pcd_trans_vec[1] += pos_stride
        elif key == 'd':
            pcd_trans_vec[1] -= pos_stride
        elif key == 'q':
            pcd_trans_vec[2] += pos_stride
        elif key == 'z':
            pcd_trans_vec[2] -= pos_stride
        elif key == '1':
            pcd_rot_vec[0] += rot_stride
        elif key == '2':
            pcd_rot_vec[0] -= rot_stride
        elif key == '3':
            pcd_rot_vec[1] += rot_stride
        elif key == '4':
            pcd_rot_vec[1] -= rot_stride
        elif key == '5':
            pcd_rot_vec[2] += rot_stride
        elif key == '6':
            pcd_rot_vec[2] -= rot_stride
        elif key == 's':
            save = True
            break
        elif key == 'c':
            break
        elif key == 'p':
            signal = 1

        # import pdb; pdb.set_trace()
        pcd_ori_world = qmult(qmult(qmult(axangle2quat([1, 0, 0], pcd_rot_vec[0]), 
            axangle2quat([0, 1, 0], pcd_rot_vec[1])), 
            axangle2quat([0, 0, 1], pcd_rot_vec[2])), 
            [1.0, 0.0, 0.0, 0.0])

        for cam_idx in tune_idx:
            cam_pos_init = camera_pose_dict[f"cam_{cam_idx}"]["position"]
            cam_ori_init = camera_pose_dict[f"cam_{cam_idx}"]["orientation"]

            cam_pos_cur = np.array(cam_pos_init) + np.array(pcd_trans_vec)
            cam_pos_cur = [float(x) for x in cam_pos_cur]

            cam_ori_cur = qmult(pcd_ori_world, cam_ori_init)
            cam_ori_cur = [float(x) for x in cam_ori_cur]
            print(f"Pos: {cam_pos_cur}\nOri: {cam_ori_cur}")
            br.sendTransform(tuple(cam_pos_cur),
                tuple([cam_ori_cur[1], cam_ori_cur[2], cam_ori_cur[3], cam_ori_cur[0]]), 
                rospy.Time.now(), f"cam{cam_idx}_link", fixed_frame)

            camera_pose_dict_new[f"cam_{cam_idx}"]["position"] = cam_pos_cur
            camera_pose_dict_new[f"cam_{cam_idx}"]["orientation"] = cam_ori_cur

            rate.sleep()

    if save:
        print(f"Final: {camera_pose_dict_new}")
        with open(os.path.join(data_dir, 'camera_pose_world.yml'), 'w') as f:
            yaml.dump(camera_pose_dict_new, f)


def get_plane_markers(pcd_msgs):
    depth_optical_frame_pose = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]
    colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]]
    
    markers = []
    for i in range(len(pcd_msgs)):
        cloud_rec = ros_numpy.point_cloud2.pointcloud2_to_array(pcd_msgs[i])
        cloud_array = cloud_rec.view('<f4').reshape(cloud_rec.shape + (-1,))
        points = cloud_array[:, :3]
        points = (quat2mat(depth_optical_frame_pose[3:]) @ points.T).T + depth_optical_frame_pose[:3]
        cam_ori = camera_pose_dict_new[f"cam_{i+1}"]["orientation"]
        points = (quat2mat(cam_ori) @ points.T).T + camera_pose_dict_new[f"cam_{i+1}"]["position"]

        x_filter = (points.T[0] > 0.4 - 0.5) & (points.T[0] < 0.4 + 0.5)
        y_filter = (points.T[1] > -0.1 - 0.5) & (points.T[1] < -0.1 + 0.5)
        z_filter = (points.T[2] > 0 - 0.1) & (points.T[2] < 0 + 0.1)
        points = points[x_filter & y_filter & z_filter]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        segment_models, inliers = pcd.segment_plane(distance_threshold=0.0075,ransac_n=3,num_iterations=100)
        [a, b, c, d] = segment_models
        # print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        plane_rot = qinverse(axangle2quat([b, -a, 0], np.arccos(c / np.linalg.norm([a, b, c]))))

        plane_marker = Marker()
        plane_marker.header.frame_id = fixed_frame
        plane_marker.header.stamp = rospy.Time.now()
        plane_marker.id = i
        plane_marker.type = plane_marker.CUBE
        plane_marker.action = plane_marker.ADD

        plane_marker.pose.position.x = 0.4
        plane_marker.pose.position.y = -0.1
        plane_marker.pose.position.z = - d / c
        plane_marker.pose.orientation.x = plane_rot[1]
        plane_marker.pose.orientation.y = plane_rot[2]
        plane_marker.pose.orientation.z = plane_rot[3]
        plane_marker.pose.orientation.w = plane_rot[0]

        plane_marker.scale.x = 0.5
        plane_marker.scale.y = 0.7
        plane_marker.scale.z = 0.005

        plane_marker.color.r = colors[i][0]
        plane_marker.color.g = colors[i][1]
        plane_marker.color.b = colors[i][2]
        plane_marker.color.a = 0.5
        
        markers.append(plane_marker)

    return markers


if __name__ == '__main__':
    main()
