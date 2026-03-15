#!/usr/bin/env python3
"""
Unified TF Publisher for Factory 5.0 Workcell v3
- All workcell static frames
- Camera optical frame correction
- FR3 Arm1 frames at home position (hardcoded, no robot_state_publisher needed)
- Dynamic box poses from Gazebo at 20Hz
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import subprocess, re, math

STATIC_FRAMES = {
    "ground_plane": ("world", 0.0, 0.0, 0.0, 0, 0, 0),
    "factory_floor": ("world", 0.0, 0.0, 0.0, 0, 0, 0),
    "work_table": ("world", 0.0, 0.3, 0.0, 0, 0, 0),
    "input_conveyor": ("world", -0.65, 0.8, 0.0, 0, 0, 0),
    "transfer_fixture": ("world", 0.0, 0.5, 0.74, 0, 0, 0),
    "output_pallet": ("world", 0.65, 0.8, 0.0, 0, 0, 0),
    "arm0_base_mount": ("world", -0.65, 0.0, 0.0, 0, 0, 0),
    "arm1_base_mount": ("world", 0.65, 0.0, 0.0, 0, 0, 0),
    "fence_rear": ("world", 0.0, 1.8, 0.0, 0, 0, 0),
    "fence_left": ("world", -1.75, 0.5, 0.0, 0, 0, 0),
    "fence_right": ("world", 1.75, 0.5, 0.0, 0, 0, 0),
    "safety_marking_front": ("world", 0.0, -1.8, 0.0, 0, 0, 0),
    "workcell_camera": ("world", 0.0, -1.2, 2.2, 0.0, 0.6, 1.5708),
    "workcell_camera/link": ("workcell_camera", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "arm0_base": ("world", -0.65, 0.0, 0.72, 0, 0, 0),
    "arm0_base_link_inertia": ("arm0_base", 0.0, 0.0, 0.0, 0, 0, 0),
    "arm0_base_link": ("arm0_base", 0.0, 0.0, 0.0, 0, 0, 0),
    "arm1_base": ("world", 0.65, 0.0, 0.72, 0, 0, math.pi),
    "arm1_fr3_link0": ("arm1_base", 0.0, 0.0, 0.0, 0, 0, 0),
    "arm1_fr3_link1": ("arm1_fr3_link0", 0.0, 0.0, 0.333, 0, 0, 0),
    "arm1_fr3_link2": ("arm1_fr3_link1", 0.0, 0.0, 0.0, 0, 0, 0),
    "arm1_fr3_link3": ("arm1_fr3_link2", 0.0, 0.316, 0.0, 0, 0, 0),
    "arm1_fr3_link4": ("arm1_fr3_link3", 0.0825, 0.0, 0.0, 0, 0, 0),
    "arm1_fr3_link5": ("arm1_fr3_link4", 0.0, 0.384, 0.0, 0, 0, 0),
    "arm1_fr3_link6": ("arm1_fr3_link5", -0.0825, 0.0, 0.0, 0, 0, 0),
    "arm1_fr3_link7": ("arm1_fr3_link6", 0.0, 0.0, 0.107, 0, 0, 0),
    "arm1_fr3_hand": ("arm1_fr3_link7", 0.0, 0.0, 0.1, 0, 0, 0),
}


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class UnifiedTFPublisher(Node):
    def __init__(self):
        super().__init__("unified_tf_publisher")
        static_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.tf_static_pub = self.create_publisher(TFMessage, "/tf_static", static_qos)
        self.tf_pub = self.create_publisher(TFMessage, "/tf", QoSProfile(depth=10))
        self.create_timer(1.0, self._publish_static)
        self.create_timer(0.05, self._publish_dynamic)
        self._publish_static()
        self.get_logger().info(
            f"✅ Unified TF Publisher v3 | {len(STATIC_FRAMES)} static frames"
        )

    def _make_tf(self, child, parent, x, y, z, roll, pitch, yaw):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
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
        for name, (parent, x, y, z, ro, pi, ya) in STATIC_FRAMES.items():
            msg.transforms.append(self._make_tf(name, parent, x, y, z, ro, pi, ya))
        optical_q = rpy_to_quat(-math.pi / 2, 0, -math.pi / 2)
        for optical_name in [
            "workcell_camera/link/color_camera",
            "workcell_camera/link/depth_camera",
        ]:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "workcell_camera/link"
            t.child_frame_id = optical_name
            t.transform.rotation.x = optical_q[0]
            t.transform.rotation.y = optical_q[1]
            t.transform.rotation.z = optical_q[2]
            t.transform.rotation.w = optical_q[3]
            msg.transforms.append(t)
        self.tf_static_pub.publish(msg)
        self.tf_pub.publish(msg)

    def _publish_dynamic(self):
        try:
            result = subprocess.run(
                [
                    "gz",
                    "topic",
                    "-e",
                    "-t",
                    "/world/factory50_workcell/dynamic_pose/info",
                    "-n",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=0.8,
            )
            poses = self._parse_gz_poses(result.stdout)
            if not poses:
                return
            msg = TFMessage()
            for name, (x, y, z, rx, ry, rz, rw) in poses.items():
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "world"
                t.child_frame_id = name
                t.transform.translation.x = x
                t.transform.translation.y = y
                t.transform.translation.z = z
                t.transform.rotation.x = rx
                t.transform.rotation.y = ry
                t.transform.rotation.z = rz
                t.transform.rotation.w = rw
                msg.transforms.append(t)
            if msg.transforms:
                self.tf_pub.publish(msg)
        except Exception:
            pass

    def _parse_gz_poses(self, text):
        poses = {}
        blocks = re.findall(r"pose\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text)
        for block in blocks:
            name_m = re.search(r'name:\s*"([^"]+)"', block)
            if not name_m:
                continue
            name = name_m.group(1)
            if name in ("link", "visual", ""):
                continue
            pos_m = re.search(r"position\s*\{([^}]*)\}", block)
            x = y = z = 0.0
            if pos_m:
                pb = pos_m.group(1)
                xm = re.search(r"x:\s*([\d\.\-e]+)", pb)
                ym = re.search(r"y:\s*([\d\.\-e]+)", pb)
                zm = re.search(r"z:\s*([\d\.\-e]+)", pb)
                if xm:
                    x = float(xm.group(1))
                if ym:
                    y = float(ym.group(1))
                if zm:
                    z = float(zm.group(1))
            ori_m = re.search(r"orientation\s*\{([^}]*)\}", block)
            rx = ry = rz = 0.0
            rw = 1.0
            if ori_m:
                ob = ori_m.group(1)
                rxm = re.search(r"x:\s*([\d\.\-e]+)", ob)
                rym = re.search(r"y:\s*([\d\.\-e]+)", ob)
                rzm = re.search(r"z:\s*([\d\.\-e]+)", ob)
                rwm = re.search(r"w:\s*([\d\.\-e]+)", ob)
                if rxm:
                    rx = float(rxm.group(1))
                if rym:
                    ry = float(rym.group(1))
                if rzm:
                    rz = float(rzm.group(1))
                if rwm:
                    rw = float(rwm.group(1))
            poses[name] = (x, y, z, rx, ry, rz, rw)
        return poses


def main():
    rclpy.init()
    rclpy.spin(UnifiedTFPublisher())


main()
