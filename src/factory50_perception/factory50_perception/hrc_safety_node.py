#!/usr/bin/env python3
"""
factory50_perception — hrc_safety_node (MediaPipe Tasks API)
Compatible with MediaPipe >= 0.10.x
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseStamped, Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
import math

ZONE_RED    = 0.5
ZONE_YELLOW = 1.2

class HRCSafetyNode(Node):
    def __init__(self):
        super().__init__('hrc_safety_node')
        self.declare_parameter('update_rate_hz', 15.0)
        self.declare_parameter('yellow_speed',    0.4)
        self.declare_parameter('red_speed',       0.0)
        self.declare_parameter('sim_mode',        False)
        self.yellow_speed  = self.get_parameter('yellow_speed').value
        self.red_speed     = self.get_parameter('red_speed').value
        self.sim_mode      = self.get_parameter('sim_mode').value
        self.bridge        = CvBridge()
        self.latest_color  = None
        self.latest_depth  = None
        self.arm0_ee_pos   = None
        self.arm1_ee_pos   = None
        self.current_zone  = 'GREEN'
        self._frame_count  = 0
        self.mp_api        = None
        self.mp_detector   = None

        if not self.sim_mode:
            self._init_mediapipe()
        else:
            self.mp_api = 'sim'
            self.get_logger().info('Running in simulation mode!')

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/camera/color/image_raw', self._color_cb, qos)
        self.create_subscription(Image, '/camera/depth/image_raw', self._depth_cb, qos)
        self.create_subscription(PoseStamped, '/arm0/ee_pose', self._arm0_cb, 10)
        self.create_subscription(PoseStamped, '/arm1/ee_pose', self._arm1_cb, 10)

        self.pub_arm0_speed = self.create_publisher(Float32, '/arm0/speed_override', 10)
        self.pub_arm1_speed = self.create_publisher(Float32, '/arm1/speed_override', 10)
        self.pub_zone       = self.create_publisher(String,  '/factory50/safety_zone', 10)
        self.pub_human      = self.create_publisher(Point,   '/factory50/human_centroid', 10)

        rate = self.get_parameter('update_rate_hz').value
        self.create_timer(1.0 / rate, self._process)
        self.get_logger().info(f'HRC Safety Node started | API: {self.mp_api}')

    def _init_mediapipe(self):
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request, os
            model_path = '/tmp/pose_landmarker.task'
            if not os.path.exists(model_path):
                self.get_logger().info('Downloading pose model...')
                url = ('https://storage.googleapis.com/mediapipe-models/'
                       'pose_landmarker/pose_landmarker_lite/float16/latest/'
                       'pose_landmarker_lite.task')
                urllib.request.urlretrieve(url, model_path)
            base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
            )
            self.mp_detector = mp_vision.PoseLandmarker.create_from_options(opts)
            self.mp_api = 'tasks'
            self.get_logger().info('MediaPipe Tasks API ready!')
        except Exception as e:
            self.get_logger().warn(f'MediaPipe init failed: {e} — using sim mode')
            self.mp_api = 'sim'

    def _color_cb(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

    def _depth_cb(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception:
            pass

    def _arm0_cb(self, msg):
        p = msg.pose.position
        self.arm0_ee_pos = np.array([p.x, p.y, p.z])

    def _arm1_cb(self, msg):
        p = msg.pose.position
        self.arm1_ee_pos = np.array([p.x, p.y, p.z])

    def _process(self):
        self._frame_count += 1
        if self.mp_api == 'sim':
            self._process_sim()
            return
        if self.latest_color is None:
            return
        cx, cy, dist_m = self._detect_pose(self.latest_color)
        if cx is None:
            self._publish_speed(1.0, 1.0, 'GREEN')
            return
        pt = Point()
        pt.x, pt.y, pt.z = float(cx), float(cy), float(dist_m)
        self.pub_human.publish(pt)
        arm0_dist = self._arm_distance(cx, cy, dist_m, self.arm0_ee_pos)
        arm1_dist = self._arm_distance(cx, cy, dist_m, self.arm1_ee_pos)
        arm0_speed, zone0 = self._zone_speed(arm0_dist)
        arm1_speed, zone1 = self._zone_speed(arm1_dist)
        self._publish_speed(arm0_speed, arm1_speed, self._worst_zone(zone0, zone1))

    def _process_sim(self):
        t      = self._frame_count * 0.1
        dist_m = 1.4 + 1.1 * math.sin(t * 0.3)
        arm0_speed, zone0 = self._zone_speed(dist_m)
        arm1_speed, zone1 = self._zone_speed(dist_m + 0.2)
        self._publish_speed(arm0_speed, arm1_speed, self._worst_zone(zone0, zone1))
        pt = Point()
        pt.x, pt.y, pt.z = 0.0, -dist_m, 0.0
        self.pub_human.publish(pt)

    def _detect_pose(self, frame_bgr):
        try:
            rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = self.mp_detector.detect(mp_image)
            if not result.pose_landmarks:
                return None, None, 1.5
            lm  = result.pose_landmarks[0]
            cx  = (lm[23].x + lm[24].x) / 2.0
            cy  = (lm[23].y + lm[24].y) / 2.0
            return cx, cy, self._estimate_depth(cx, cy)
        except Exception:
            return None, None, 1.5

    def _estimate_depth(self, cx_norm, cy_norm):
        if self.latest_depth is None:
            return 1.5
        h, w  = self.latest_depth.shape
        px    = int(np.clip(cx_norm * w, 0, w-1))
        py    = int(np.clip(cy_norm * h, 0, h-1))
        patch = self.latest_depth[max(0,py-3):min(h,py+4), max(0,px-3):min(w,px+4)]
        valid = patch[patch > 0.1]
        return float(np.median(valid)) if len(valid) > 0 else 1.5

    def _arm_distance(self, cx_norm, cy_norm, depth_m, arm_ee):
        if arm_ee is None:
            return 99.0
        human_pos = np.array([(cx_norm-0.5)*depth_m, (cy_norm-0.5)*depth_m, depth_m])
        return float(np.linalg.norm(human_pos - arm_ee))

    def _zone_speed(self, distance_m):
        if distance_m < ZONE_RED:
            return self.red_speed, 'RED'
        elif distance_m < ZONE_YELLOW:
            return self.yellow_speed, 'YELLOW'
        return 1.0, 'GREEN'

    def _worst_zone(self, z1, z2):
        priority = {'RED': 0, 'YELLOW': 1, 'GREEN': 2}
        return z1 if priority[z1] <= priority[z2] else z2

    def _publish_speed(self, arm0_speed, arm1_speed, zone):
        self.pub_arm0_speed.publish(Float32(data=float(arm0_speed)))
        self.pub_arm1_speed.publish(Float32(data=float(arm1_speed)))
        self.pub_zone.publish(String(data=zone))
        if zone != self.current_zone:
            icons = {'GREEN': '🟢', 'YELLOW': '🟡', 'RED': '🔴'}
            self.get_logger().info(
                f'{icons[zone]} {self.current_zone}→{zone} | '
                f'Arm0:{arm0_speed:.0%} Arm1:{arm1_speed:.0%}')
            self.current_zone = zone

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
