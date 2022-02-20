# 1st terminal
conda deactivate
roslaunch RoboCook_ROS visualize.launch

# 2nd terminal
conda activate polymetis
python /scr/hxu/catkin_ws/src/RoboCook_ROS/src/rviz_viewer.py