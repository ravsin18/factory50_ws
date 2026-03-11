#!/usr/bin/env python3
"""
Single unified TF publisher for Factory 5.0 workcell.
Publishes ALL static + dynamic transforms from one node.
Fixes QoS conflicts from multiple static_transform_publisher instances.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import subprocess, re, math

# ── Static model positions ────────────────────────────────────────────────────
STATIC_FRAMES = {
    # Workcell objects
    "ground_plane":             (0.0,   0.0,  0.0,  0, 0, 0),
    "factory_floor":            (0.0,   0.0,  0.0,  0, 0, 0),
    "work_table":               (0.0,   0.3,  0.0,  0, 0, 0),
    "input_conveyor":           (-0.65, 0.8,  0.0,  0, 0, 0),
    "transfer_fixture":         (0.0,   0.5,  0.74, 0, 0, 0),
    "output_pallet":            (0.65,  0.8,  0.0,  0, 0, 0),
    "arm0_base_mount":          (-0.65, 0.0,  0.0,  0, 0, 0),
    "arm1_base_mount":          (0.65,  0.0,  0.0,  0, 0, 0),
    "fence_rear":               (0.0,   1.8,  0.0,  0, 0, 0),
    "fence_left":               (-1.75, 0.5,  0.0,  0, 0, 0),
    "fence_right":              (1.75,  0.5,  0.0,  0, 0, 0),
    "safety_marking_front":     (0.0,  -1.8,  0.0,  0, 0, 0),
    # Camera chain — world → workcell_camera → link → sensors
    "workcell_camera":          (0.0,  -1.2,  2.2,  0, 0.6, 1.5708),
    "workcell_camera/link":     (0.0,   0.0,  0.0,  0, 0, 0),
    "workcell_camera/link/color_camera": (0.0, 0.0, 0.0, 0, 0, 0),
    "workcell_camera/link/depth_camera": (0.0, 0.0, 0.0, 0, 0, 0),
}

# Parent frame for each — most are children of world,
# camera sub-frames have specific parents
FRAME_PARENTS = {
    "workcell_camera/link":              "workcell_camera",
    "workcell_camera/link/color_camera": "workcell_camera/link",
    "workcell_camera/link/depth_camera": "workcell_camera/link",
}

def rpy_to_quat(roll, pitch, yaw):
    """Convert roll/pitch/yaw to quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return (
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy,
    )

class UnifiedTFPublisher(Node):
    def __init__(self):
        super().__init__('unified_tf_publisher')

        # ── TRANSIENT_LOCAL QoS for /tf_static (required!) ────────────────────
        static_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        dynamic_qos = QoSProfile(depth=10)

        self.tf_static_pub = self.create_publisher(
            TFMessage, '/tf_static', static_qos)
        self.tf_pub = self.create_publisher(
            TFMessage, '/tf', dynamic_qos)

        # Publish static frames immediately + every 2s for late subscribers
        self.create_timer(2.0,  self._publish_static)
        # Publish dynamic frames at 20 Hz
        self.create_timer(0.05, self._publish_dynamic)

        self._publish_static()  # publish once immediately on startup
        self.get_logger().info('✅ Unified TF Publisher started!')

    def _make_tf(self, child, parent, x, y, z, roll, pitch, yaw):
        t = TransformStamped()
        t.header.stamp       = self.get_clock().now().to_msg()
        t.header.frame_id    = parent
        t.child_frame_id     = child
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        return t

    def _publish_static(self):
        msg = TFMessage()
        for name, (x, y, z, ro, pi, ya) in STATIC_FRAMES.items():
            parent = FRAME_PARENTS.get(name, 'world')
            msg.transforms.append(
                self._make_tf(name, parent, x, y, z, ro, pi, ya))
        self.tf_static_pub.publish(msg)
        self.tf_pub.publish(msg)
        self.get_logger().info(
            f'Published {len(msg.transforms)} static frames to /tf_static')

    def _publish_dynamic(self):
        try:
            result = subprocess.run(
                ['gz', 'topic', '-e', '-t',
                 '/world/factory50_workcell/dynamic_pose/info', '-n', '1'],
                capture_output=True, text=True, timeout=0.8
            )
            poses = self._parse_gz_poses(result.stdout)
            if not poses:
                return
            msg = TFMessage()
            for name, (x, y, z, rx, ry, rz, rw) in poses.items():
                t = TransformStamped()
                t.header.stamp       = self.get_clock().now().to_msg()
                t.header.frame_id    = 'world'
                t.child_frame_id     = name
                t.transform.translation.x = x
                t.transform.translation.y = y
                t.transform.translation.z = z
                t.transform.rotation.x    = rx
                t.transform.rotation.y    = ry
                t.transform.rotation.z    = rz
                t.transform.rotation.w    = rw
                msg.transforms.append(t)
            if msg.transforms:
                self.tf_pub.publish(msg)
        except Exception:
            pass

    def _parse_gz_poses(self, text):
        poses = {}
        blocks = re.findall(
            r'pose\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
        for block in blocks:
            name_m = re.search(r'name:\s*"([^"]+)"', block)
            if not name_m:
                continue
            name = name_m.group(1)
            if name in ('link', 'visual', ''):
                continue
            pos_m = re.search(r'position\s*\{([^}]*)\}', block)
            x = y = z = 0.0
            if pos_m:
                pb = pos_m.group(1)
                xm = re.search(r'x:\s*([\d\.\-e]+)', pb)
                ym = re.search(r'y:\s*([\d\.\-e]+)', pb)
                zm = re.search(r'z:\s*([\d\.\-e]+)', pb)
                if xm: x = float(xm.group(1))
                if ym: y = float(ym.group(1))
                if zm: z = float(zm.group(1))
            ori_m = re.search(r'orientation\s*\{([^}]*)\}', block)
            rx = ry = rz = 0.0; rw = 1.0
            if ori_m:
                ob = ori_m.group(1)
                rxm = re.search(r'x:\s*([\d\.\-e]+)', ob)
                rym = re.search(r'y:\s*([\d\.\-e]+)', ob)
                rzm = re.search(r'z:\s*([\d\.\-e]+)', ob)
                rwm = re.search(r'w:\s*([\d\.\-e]+)', ob)
                if rxm: rx = float(rxm.group(1))
                if rym: ry = float(rym.group(1))
                if rzm: rz = float(rzm.group(1))
                if rwm: rw = float(rwm.group(1))
            poses[name] = (x, y, z, rx, ry, rz, rw)
        return poses

def main():
    rclpy.init()
    rclpy.spin(UnifiedTFPublisher())

main()
