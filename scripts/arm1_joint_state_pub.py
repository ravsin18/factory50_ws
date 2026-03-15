#!/usr/bin/env python3
"""
Publishes static joint states for Franka FR3 Arm1.
Keeps arm in upright home position for visualization.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

# FR3 home position — arm upright
JOINT_NAMES = [
    'arm1_fr3_joint1',
    'arm1_fr3_joint2',
    'arm1_fr3_joint3',
    'arm1_fr3_joint4',
    'arm1_fr3_joint5',
    'arm1_fr3_joint6',
    'arm1_fr3_joint7',
]

HOME_POSITIONS = [
    0.0,           # joint1
    -0.785,        # joint2  (-45°)
    0.0,           # joint3
    -2.356,        # joint4  (-135°)
    0.0,           # joint5
    1.571,         # joint6  (+90°)
    0.785,         # joint7  (+45°)
]

class Arm1JointStatePublisher(Node):
    def __init__(self):
        super().__init__('arm1_joint_state_publisher')
        self.pub = self.create_publisher(
            JointState, '/arm1/joint_states', 10)
        self.create_timer(0.05, self._publish)  # 20 Hz
        self.get_logger().info('✅ Arm1 joint state publisher started!')

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = JOINT_NAMES
        msg.position     = HOME_POSITIONS
        msg.velocity     = [0.0] * len(JOINT_NAMES)
        msg.effort       = [0.0] * len(JOINT_NAMES)
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(Arm1JointStatePublisher())

main()
