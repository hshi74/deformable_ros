# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import rospy

from polymetis import RobotInterface, GripperInterface
from sensor_msgs.msg import JointState


def get_joint_states(robot, gripper):
    joint_pos = robot.get_joint_angles()
    joint_vel = robot.get_joint_velocities()
    gripper_state = gripper.get_state()
    return joint_pos.tolist(), joint_vel.tolist(), gripper_state.width


def main():
    rospy.init_node('joint_state_publisher', anonymous=True)

    # Initialize robot interface
    robot = RobotInterface(
        ip_address="192.168.0.2",
        enforce_version=False,
    )

    gripper = GripperInterface(
        ip_address="192.168.0.2",
    )

    joint_pos_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    joint_names = [f'panda_joint{i}' for i in range(1, 8)]
    joint_names += ['panda_finger_joint1', 'panda_finger_joint1']
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        joint_pos, joint_vel, gripper_state = get_joint_states(robot, gripper)
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = joint_names
        joint_state.position = joint_pos + [gripper_state * 0.5, gripper_state * 0.5]
        joint_state.velocity = joint_vel + [0.0, 0.0]
        # import pdb; pdb.set_trace()
        joint_pos_pub.publish(joint_state)
        rate.sleep()


if __name__ == "__main__":
    main()
