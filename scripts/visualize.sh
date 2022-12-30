# 1st terminal
conda deactivate
roslaunch deformable_ros visualize.launch

# 2nd terminal
conda activate polymetis
python /scr/hxu/catkin_ws/src/deformable_ros/src/rviz_viewer.py