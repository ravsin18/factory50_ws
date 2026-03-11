#!/usr/bin/env python3
"""
factory50_twin — opcua_bridge_node
Bridges ROS2 topics → OPC-UA server for Digital Twin.

Streams live robot data from both arms, HRC safety state,
and MARL agent metrics to InfluxDB and Foxglove.

OPC-UA Server: opc.tcp://localhost:4840/factory50/
InfluxDB:      http://localhost:8086  (bucket: robot_metrics)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

import asyncio
import threading
import time
from datetime import datetime, timezone

from asyncua import Server, ua
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


# ── InfluxDB Config (matches docker-compose.yml) ──────────────────────────────
INFLUX_URL   = "http://localhost:8086"
INFLUX_TOKEN = "factory50-super-token-2024"
INFLUX_ORG   = "factory50_org"
INFLUX_BUCKET= "robot_metrics"


class OPCUABridgeNode(Node):

    def __init__(self):
        super().__init__('opcua_bridge_node')

        # ── State ─────────────────────────────────────────────────────────────
        self.data = {
            'arm0_joints':      [0.0] * 6,
            'arm1_joints':      [0.0] * 7,
            'arm0_eef_x':       0.0,
            'arm0_eef_y':       0.0,
            'arm0_eef_z':       0.0,
            'arm1_eef_x':       0.0,
            'arm1_eef_y':       0.0,
            'arm1_eef_z':       0.0,
            'arm0_speed':       1.0,
            'arm1_speed':       1.0,
            'safety_zone':      'GREEN',
            'human_dist_arm0':  99.0,
            'human_dist_arm1':  99.0,
            'task_phase':       0,
            'cycle_time':       0.0,
            'boxes_packed':     0,
        }
        self._cycle_start = time.time()
        self._lock        = threading.Lock()

        # ── ROS2 Subscribers ──────────────────────────────────────────────────
        self.create_subscription(JointState,  '/arm0/joint_states',    self._arm0_joints_cb, 10)
        self.create_subscription(JointState,  '/arm1/joint_states',    self._arm1_joints_cb, 10)
        self.create_subscription(PoseStamped, '/arm0/ee_pose',         self._arm0_eef_cb,    10)
        self.create_subscription(PoseStamped, '/arm1/ee_pose',         self._arm1_eef_cb,    10)
        self.create_subscription(Float32,     '/arm0/speed_override',  self._arm0_speed_cb,  10)
        self.create_subscription(Float32,     '/arm1/speed_override',  self._arm1_speed_cb,  10)
        self.create_subscription(String,      '/factory50/safety_zone',self._zone_cb,        10)

        # ── InfluxDB Client ───────────────────────────────────────────────────
        try:
            self._influx = InfluxDBClient(
                url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            self._write  = self._influx.write_api(write_options=SYNCHRONOUS)
            self.get_logger().info('InfluxDB connected ✓')
        except Exception as e:
            self._write = None
            self.get_logger().warn(f'InfluxDB unavailable: {e}')

        # ── Start OPC-UA server in background thread ───────────────────────────
        self._opcua_thread = threading.Thread(
            target=self._run_opcua, daemon=True)
        self._opcua_thread.start()

        # ── Periodic InfluxDB write timer (1 Hz) ──────────────────────────────
        self.create_timer(1.0, self._write_to_influx)

        self.get_logger().info('OPC-UA Bridge Node started')
        self.get_logger().info('OPC-UA:  opc.tcp://localhost:4840/factory50/')
        self.get_logger().info('Grafana: http://localhost:3000')

    # ── ROS2 Callbacks ────────────────────────────────────────────────────────
    def _arm0_joints_cb(self, msg):
        with self._lock:
            self.data['arm0_joints'] = list(msg.position[:6])

    def _arm1_joints_cb(self, msg):
        with self._lock:
            self.data['arm1_joints'] = list(msg.position[:7])

    def _arm0_eef_cb(self, msg):
        with self._lock:
            p = msg.pose.position
            self.data['arm0_eef_x'] = p.x
            self.data['arm0_eef_y'] = p.y
            self.data['arm0_eef_z'] = p.z

    def _arm1_eef_cb(self, msg):
        with self._lock:
            p = msg.pose.position
            self.data['arm1_eef_x'] = p.x
            self.data['arm1_eef_y'] = p.y
            self.data['arm1_eef_z'] = p.z

    def _arm0_speed_cb(self, msg):
        with self._lock:
            self.data['arm0_speed'] = msg.data

    def _arm1_speed_cb(self, msg):
        with self._lock:
            self.data['arm1_speed'] = msg.data

    def _zone_cb(self, msg):
        with self._lock:
            prev = self.data['safety_zone']
            self.data['safety_zone'] = msg.data
            # Track cycle time when box is packed
            if prev == 'GREEN' and msg.data == 'RED':
                self.data['cycle_time'] = time.time() - self._cycle_start

    # ── InfluxDB Write ────────────────────────────────────────────────────────
    def _write_to_influx(self):
        if self._write is None:
            return
        with self._lock:
            d = dict(self.data)

        now = datetime.now(timezone.utc)
        try:
            points = [
                Point("robot_state")
                    .field("arm0_speed",      d['arm0_speed'])
                    .field("arm1_speed",      d['arm1_speed'])
                    .field("arm0_eef_x",      d['arm0_eef_x'])
                    .field("arm0_eef_y",      d['arm0_eef_y'])
                    .field("arm0_eef_z",      d['arm0_eef_z'])
                    .field("arm1_eef_x",      d['arm1_eef_x'])
                    .field("arm1_eef_y",      d['arm1_eef_y'])
                    .field("arm1_eef_z",      d['arm1_eef_z'])
                    .time(now),

                Point("safety_metrics")
                    .field("safety_zone",     d['safety_zone'])
                    .field("human_dist_arm0", d['human_dist_arm0'])
                    .field("human_dist_arm1", d['human_dist_arm1'])
                    .time(now),

                Point("production_kpis")
                    .field("task_phase",      float(d['task_phase']))
                    .field("cycle_time",      d['cycle_time'])
                    .field("boxes_packed",    float(d['boxes_packed']))
                    .time(now),
            ]
            self._write.write(bucket=INFLUX_BUCKET, record=points)
        except Exception as e:
            self.get_logger().debug(f'InfluxDB write error: {e}')

    # ── OPC-UA Server ─────────────────────────────────────────────────────────
    def _run_opcua(self):
        asyncio.run(self._opcua_server())

    async def _opcua_server(self):
        server = Server()
        await server.init()
        server.set_endpoint("opc.tcp://0.0.0.0:4840/factory50/")
        server.set_server_name("Factory 5.0 Digital Twin")

        uri   = "http://factory50.de/digitalTwin"
        idx   = await server.register_namespace(uri)
        root  = server.get_objects_node()

        # ── Create OPC-UA node structure ──────────────────────────────────────
        cell  = await root.add_object(idx, "WorkCell")

        # Arm 0 (UR5e)
        arm0_node = await cell.add_object(idx, "Arm0_UR5e")
        opc_arm0_speed = await arm0_node.add_variable(idx, "SpeedOverride", 1.0)
        opc_arm0_eef_x = await arm0_node.add_variable(idx, "EEF_X", 0.0)
        opc_arm0_eef_y = await arm0_node.add_variable(idx, "EEF_Y", 0.0)
        opc_arm0_eef_z = await arm0_node.add_variable(idx, "EEF_Z", 0.0)
        await opc_arm0_speed.set_writable()

        # Arm 1 (Franka)
        arm1_node = await cell.add_object(idx, "Arm1_Franka")
        opc_arm1_speed = await arm1_node.add_variable(idx, "SpeedOverride", 1.0)
        opc_arm1_eef_x = await arm1_node.add_variable(idx, "EEF_X", 0.0)
        opc_arm1_eef_y = await arm1_node.add_variable(idx, "EEF_Y", 0.0)
        opc_arm1_eef_z = await arm1_node.add_variable(idx, "EEF_Z", 0.0)

        # Safety
        safety_node    = await cell.add_object(idx, "Safety")
        opc_zone       = await safety_node.add_variable(idx, "Zone", "GREEN")
        opc_h_dist0    = await safety_node.add_variable(idx, "HumanDist_Arm0", 99.0)
        opc_h_dist1    = await safety_node.add_variable(idx, "HumanDist_Arm1", 99.0)

        # KPIs
        kpi_node       = await cell.add_object(idx, "KPIs")
        opc_phase      = await kpi_node.add_variable(idx, "TaskPhase",   0)
        opc_cycle      = await kpi_node.add_variable(idx, "CycleTime",   0.0)
        opc_boxes      = await kpi_node.add_variable(idx, "BoxesPacked", 0)

        self.get_logger().info('OPC-UA server running: opc.tcp://localhost:4840/factory50/')

        async with server:
            while True:
                with self._lock:
                    d = dict(self.data)

                # Update all OPC-UA nodes
                await opc_arm0_speed.write_value(float(d['arm0_speed']))
                await opc_arm0_eef_x.write_value(float(d['arm0_eef_x']))
                await opc_arm0_eef_y.write_value(float(d['arm0_eef_y']))
                await opc_arm0_eef_z.write_value(float(d['arm0_eef_z']))
                await opc_arm1_speed.write_value(float(d['arm1_speed']))
                await opc_arm1_eef_x.write_value(float(d['arm1_eef_x']))
                await opc_arm1_eef_y.write_value(float(d['arm1_eef_y']))
                await opc_arm1_eef_z.write_value(float(d['arm1_eef_z']))
                await opc_zone.write_value(str(d['safety_zone']))
                await opc_h_dist0.write_value(float(d['human_dist_arm0']))
                await opc_h_dist1.write_value(float(d['human_dist_arm1']))
                await opc_phase.write_value(int(d['task_phase']))
                await opc_cycle.write_value(float(d['cycle_time']))
                await opc_boxes.write_value(int(d['boxes_packed']))

                await asyncio.sleep(0.01)  # 100 Hz OPC-UA update


def main(args=None):
    rclpy.init(args=args)
    node = OPCUABridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
