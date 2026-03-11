#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Factory 5.0 — Master Startup Script
# Run this once after container starts:  bash start_workcell.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

WS=/workspaces/factory50_ws

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   🏭 Factory 5.0 Workcell Starting...        ║"
echo "╚══════════════════════════════════════════════╝"

# Source ROS2
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash 2>/dev/null || true

# Kill any leftover processes from previous session
echo "[0/4] Cleaning up old processes..."
pkill -f "gz sim"           2>/dev/null || true
pkill -f parameter_bridge   2>/dev/null || true
pkill -f all_transforms     2>/dev/null || true
pkill -f foxglove_bridge    2>/dev/null || true
pkill -f static_transform   2>/dev/null || true
sleep 2

# 1. Gazebo headless
echo "[1/4] Starting Gazebo Sim (headless)..."
gz sim -s -r $WS/src/factory50_simulation/worlds/workcell.sdf &
sleep 5

# 2. ROS-Gazebo bridge
echo "[2/4] Starting ROS-Gazebo bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock \
  /camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image \
  /camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image \
  "/camera/depth/image_raw/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked" \
  /camera/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo \
  /camera/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo &
sleep 3

# 3. Unified TF publisher (static + dynamic transforms)
echo "[3/4] Starting TF publisher..."
python3 $WS/scripts/all_transforms.py &
sleep 3

# 4. Foxglove WebSocket bridge
echo "[4/4] Starting Foxglove bridge..."
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 &
sleep 2

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   ✅ Factory 5.0 Workcell is RUNNING!        ║"
echo "║                                              ║"
echo "║   Open:    https://app.foxglove.dev          ║"
echo "║   Connect: ws://localhost:8765               ║"
echo "║                                              ║"
echo "║   Active topics:                             ║"
echo "║   /camera/color/image_raw      (30 Hz)       ║"
echo "║   /camera/depth/image_raw      (30 Hz)       ║"
echo "║   /camera/depth/image_raw/points (30 Hz)     ║"
echo "║   /tf + /tf_static             (20 Hz)       ║"
echo "║   /clock                                     ║"
echo "║                                              ║"
echo "║   Stop all:  bash stop_workcell.sh           ║"
echo "╚══════════════════════════════════════════════╝"
