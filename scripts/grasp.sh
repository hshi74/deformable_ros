# Terminal 1
roslaunch realsense2_camera rs_multiple_devices.launch

# Terminal 2
python /scr/hxu/catkin_ws/src/RoboCook_ROS/src/get_point_cloud.py

# Terminal 3
conda activate polymetis
python /scr/hxu/catkin_ws/src/RoboCook_ROS/src/grasp.py