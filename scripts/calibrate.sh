# 1st terminal
gnome-terminal -- python /scr/hxu/catkin_ws/src/RoboCook_ROS/src/get_cam_pose.py

# 2nd terminal

declare -a arr=("1" "2" "3" "4")
declare -a arr2=("125322063608" "125322060645" "125322061326" "123122060454")
for i in "${arr[@]}"
do
    gnome-terminal -- roslaunch RoboCook_ROS ar_track.launch cam_idx:=$i serial_no:=${arr2[$i]}
done

