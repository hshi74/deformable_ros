# 1st terminal
conda deactivate
roslaunch robocook_ros visualize.launch

# 2nd terminal
conda activate polymetis
python /scr/hxu/catkin_ws/src/robocook_ros/src/rviz_viewer.py