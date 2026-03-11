#!/usr/bin/env python3
"""
factory50_perception — hrc_safety_node
Human-Robot Collaboration safety node using MediaPipe pose detection.

Detects human skeleton from depth camera, computes distance to each arm,
and publishes speed override commands per ISO 10218 zone logic.

Topics Published:
  /factory50/human_state        (factory50_msgs/HumanState)
  /arm1/speed_override          (std_msgs/Float32)   0.0–1.0
  /arm2/speed_override          (std_msgs/Float32)   0.0–1.0
  /factory50/safety_zone        (std_msgs/String)    GREEN/YELLOW/RED

Topics Subscribed:
  /camera/depth/image_raw       (sensor_msgs/Image)
  /camera/color/image_raw       (sensor_msgs/Image)
  /arm1/ee_pose                 (geometry_msgs/PoseStamped)
  /arm2/ee_pose                 (geometry_msgs/PoseStamped)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseStamped, Point
from cv_bridge import CvBridge

import cv2
import mediapipe as mp
import numpy as np


# ── ISO 10218 Safety Zone Thresholds (metres) ─────────────────────────────────
ZONE_RED    = 0.5   # < 0.5m  → immediate stop
ZONE_YELLOW = 1.2   # < 1.2m  → reduced speed
# > 1.2m = GREEN → full speed


class HRCSafetyNode(Node):

    def __init__(self):
        super().__init__('hrc_safety_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('update_rate_hz',   30.0)
        self.declare_parameter('yellow_speed',      0.4)   # 40% speed in yellow zone
        self.declare_parameter('red_speed',         0.0)   # full stop in red zone
        self.declare_parameter('camera_fov_deg',   69.0)   # RealSense D435i default
        self.declare_parameter('debug_viz',        True)

        self.yellow_speed  = self.get_parameter('yellow_speed').value
        self.red_speed     = self.get_parameter('red_speed').value
        self.debug_viz     = self.get_parameter('debug_viz').value

        # ── MediaPipe Setup ───────────────────────────────────────────────────
        self.mp_pose     = mp.solutions.pose
        self.mp_drawing  = mp.solutions.drawing_utils
        self.pose        = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1           # 0=lite, 1=full, 2=heavy — use 1 for 4GB GPU
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.bridge          = CvBridge()
        self.latest_color    = None
        self.latest_depth    = None
        self.arm1_ee_pos     = None
        self.arm2_ee_pos     = None
        self.current_zone    = "GREEN"

        # ── QoS ───────────────────────────────────────────────────────────────
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Image, '/camera/color/image_raw',
                                 self._color_cb, qos)
        self.create_subscription(Image, '/camera/depth/image_raw',
                                 self._depth_cb, qos)
        self.create_subscription(PoseStamped, '/arm1/ee_pose',
                                 self._arm1_cb, 10)
        self.create_subscription(PoseStamped, '/arm2/ee_pose',
                                 self._arm2_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_arm1_speed  = self.create_publisher(Float32, '/arm1/speed_override', 10)
        self.pub_arm2_speed  = self.create_publisher(Float32, '/arm2/speed_override', 10)
        self.pub_zone        = self.create_publisher(String,  '/factory50/safety_zone', 10)
        self.pub_human_pos   = self.create_publisher(Point,   '/factory50/human_centroid', 10)

        # ── Main Timer ────────────────────────────────────────────────────────
        rate = self.get_parameter('update_rate_hz').value
        self.create_timer(1.0 / rate, self._process)

        self.get_logger().info(
            f'HRC Safety Node started | '
            f'RED < {ZONE_RED}m | YELLOW < {ZONE_YELLOW}m'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _color_cb(self, msg):
        self.latest_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')

    def _arm1_cb(self, msg):
        p = msg.pose.position
        self.arm1_ee_pos = np.array([p.x, p.y, p.z])

    def _arm2_cb(self, msg):
        p = msg.pose.position
        self.arm2_ee_pos = np.array([p.x, p.y, p.z])

    # ── Main Processing Loop ──────────────────────────────────────────────────
    def _process(self):
        if self.latest_color is None:
            return

        # Run MediaPipe pose detection
        rgb = cv2.cvtColor(self.latest_color, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            # No human detected — full speed allowed
            self._publish_speed(1.0, 1.0, "GREEN")
            return

        # Get human centroid (average of hip landmarks)
        lm  = results.pose_landmarks.landmark
        cx  = (lm[self.mp_pose.PoseLandmark.LEFT_HIP].x +
               lm[self.mp_pose.PoseLandmark.RIGHT_HIP].x) / 2.0
        cy  = (lm[self.mp_pose.PoseLandmark.LEFT_HIP].y +
               lm[self.mp_pose.PoseLandmark.RIGHT_HIP].y) / 2.0

        # Estimate depth from depth image if available
        dist_m = self._estimate_depth(cx, cy)

        # Publish human centroid
        pt = Point()
        pt.x, pt.y, pt.z = float(cx), float(cy), float(dist_m)
        self.pub_human_pos.publish(pt)

        # Determine zone and compute speed for each arm
        arm1_dist = self._arm_distance(cx, cy, dist_m, self.arm1_ee_pos)
        arm2_dist = self._arm_distance(cx, cy, dist_m, self.arm2_ee_pos)

        arm1_speed, zone1 = self._zone_speed(arm1_dist)
        arm2_speed, zone2 = self._zone_speed(arm2_dist)

        # Worst zone across both arms drives the global zone
        global_zone = self._worst_zone(zone1, zone2)
        self._publish_speed(arm1_speed, arm2_speed, global_zone)

        # Debug visualization
        if self.debug_viz:
            self._draw_debug(rgb, results, arm1_dist, arm2_dist, global_zone)

    def _estimate_depth(self, cx_norm, cy_norm):
        """Estimate human distance from depth image pixel."""
        if self.latest_depth is None:
            return 1.5  # fallback: assume 1.5m

        h, w = self.latest_depth.shape
        px = int(cx_norm * w)
        py = int(cy_norm * h)
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)

        # Sample 5×5 patch and take median (robust to noise)
        patch = self.latest_depth[
            max(0, py-2):min(h, py+3),
            max(0, px-2):min(w, px+3)
        ]
        valid = patch[patch > 0.1]
        return float(np.median(valid)) if len(valid) > 0 else 1.5

    def _arm_distance(self, cx_norm, cy_norm, depth_m, arm_ee):
        """Compute Euclidean distance from human centroid to arm end-effector."""
        if arm_ee is None:
            return 99.0  # arm pose not available — assume safe

        # Approximate human 3D position from normalised camera coords
        # (simplified — replace with proper camera intrinsics for accuracy)
        human_pos = np.array([
            (cx_norm - 0.5) * depth_m,
            (cy_norm - 0.5) * depth_m,
            depth_m
        ])
        return float(np.linalg.norm(human_pos - arm_ee))

    def _zone_speed(self, distance_m):
        """Map distance to ISO 10218 zone and speed multiplier."""
        if distance_m < ZONE_RED:
            return self.red_speed, "RED"
        elif distance_m < ZONE_YELLOW:
            return self.yellow_speed, "YELLOW"
        else:
            return 1.0, "GREEN"

    def _worst_zone(self, z1, z2):
        priority = {"RED": 0, "YELLOW": 1, "GREEN": 2}
        return z1 if priority[z1] <= priority[z2] else z2

    def _publish_speed(self, arm1_speed, arm2_speed, zone):
        """Publish speed overrides and zone state."""
        self.pub_arm1_speed.publish(Float32(data=float(arm1_speed)))
        self.pub_arm2_speed.publish(Float32(data=float(arm2_speed)))

        if zone != self.current_zone:
            self.get_logger().info(
                f'Safety zone: {self.current_zone} → {zone} | '
                f'Arm1: {arm1_speed:.0%}  Arm2: {arm2_speed:.0%}'
            )
            self.current_zone = zone

        self.pub_zone.publish(String(data=zone))

    def _draw_debug(self, rgb, results, d1, d2, zone):
        """Draw skeleton + zone info on debug window."""
        viz = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            viz, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        color = {"GREEN": (0,255,0), "YELLOW": (0,255,255), "RED": (0,0,255)}
        cv2.putText(viz, f'Zone: {zone}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color[zone], 2)
        cv2.putText(viz, f'Arm1: {d1:.2f}m  Arm2: {d2:.2f}m', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow('HRC Safety Monitor', viz)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = HRCSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
