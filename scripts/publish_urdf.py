#!/usr/bin/env python3
"""
Publishes robot_description with TRANSIENT_LOCAL QoS
so Foxglove can load the URDF for 3D visualization.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String

class URDFPublisher(Node):
    def __init__(self):
        super().__init__('urdf_publisher')

        # TRANSIENT_LOCAL = latched — Foxglove receives it even if it connects late
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.pub = self.create_publisher(String, '/robot_description', qos)

        # Read URDF from file
        with open('/tmp/ur5e.urdf', 'r') as f:
            urdf = f.read()

        msg = String()
        msg.data = urdf
        self.pub.publish(msg)
        self.get_logger().info('✅ URDF published with TRANSIENT_LOCAL QoS!')

        # Keep publishing every 2s for any late subscribers
        self.create_timer(2.0, lambda: self.pub.publish(msg))

def main():
    rclpy.init()
    rclpy.spin(URDFPublisher())

main()
