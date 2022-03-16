# Terminal 1
roslaunch realsense2_camera rs_multiple_devices.launch

# Terminal 2
python /scr/hxu/catkin_ws/src/robocook_ros/src/get_point_cloud.py

# Terminal 3
conda activate polymetis
python /scr/hxu/catkin_ws/src/robocook_ros/src/grasp.py