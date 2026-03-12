#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Factory 5.0 — Master Startup Script v2
# Includes: Gazebo + Bridges + TF + UR5e + Foxglove
# Usage: bash start_workcell.sh
# ─────────────────────────────────────────────────────────────────────────────
WS=/workspaces/factory50_ws

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   🏭 Factory 5.0 Workcell Starting...  v2   ║"
echo "╚══════════════════════════════════════════════╝"

# Source ROS2
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash 2>/dev/null || true

# Kill leftover processes
echo "[0/6] Cleaning up..."
pkill -f "gz sim"           2>/dev/null || true
pkill -f parameter_bridge   2>/dev/null || true
pkill -f all_transforms     2>/dev/null || true
pkill -f foxglove_bridge    2>/dev/null || true
pkill -f robot_state_pub    2>/dev/null || true
pkill -f static_transform   2>/dev/null || true
sleep 2

# 1. Gazebo headless
echo "[1/6] Starting Gazebo..."
gz sim -s -r $WS/src/factory50_simulation/worlds/workcell.sdf &
sleep 5

# 2. Generate UR5e URDF
echo "[2/6] Generating UR5e URDF..."
xacro $(ros2 pkg prefix ur_description)/share/ur_description/urdf/ur.urdf.xacro \
  ur_type:=ur5e \
  name:=ur5e \
  tf_prefix:=arm0_ \
  force_abs_paths:=true \
  > /tmp/ur5e.urdf
echo "    UR5e URDF generated ($(wc -l < /tmp/ur5e.urdf) lines)"

# 3. ROS-Gazebo bridge
echo "[3/6] Starting ROS-Gazebo bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock \
  /camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image \
  /camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image \
  "/camera/depth/image_raw/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked" \
  /camera/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo \
  /camera/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo \
  "/model/ur5e_arm0/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model" &
sleep 3

# 4. Spawn UR5e into Gazebo
echo "[4/6] Spawning UR5e arm..."
ros2 run ros_gz_sim create \
  -name ur5e_arm0 \
  -file /tmp/ur5e.urdf \
  -x -0.65 -y 0.0 -z 0.72 \
  -R 0 -P 0 -Y 0
sleep 2

# 5. Robot state publisher (joints → TF)
echo "[5/6] Starting robot state publisher..."
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args \
  -p robot_description:="$(cat /tmp/ur5e.urdf)" \
  -p tf_prefix:=arm0_ &
sleep 2

# 5b. Unified TF publisher (workcell + camera frames)
python3 $WS/scripts/all_transforms.py &
sleep 2

# 6. Foxglove bridge
echo "[6/6] Starting Foxglove bridge..."
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 &
sleep 2

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   ✅ Factory 5.0 Workcell RUNNING!  v2       ║"
echo "║                                              ║"
echo "║   Foxglove: https://app.foxglove.dev         ║"
echo "║   Connect:  ws://localhost:8765              ║"
echo "║                                              ║"
echo "║   Arms:     UR5e spawned at (-0.65, 0, 0.72) ║"
echo "║   Topics:   /model/ur5e_arm0/joint_state     ║"
echo "║             /tf  /tf_static  /clock          ║"
echo "╚══════════════════════════════════════════════╝"

# 7. UR5e mock hardware driver
echo "[7/7] Starting UR5e driver..."
ros2 launch ur_robot_driver ur5e.launch.py \
  robot_ip:=192.168.0.1 \
  use_mock_hardware:=true \
  mock_sensor_commands:=true \
  launch_rviz:=false \
  tf_prefix:=arm0_ &
sleep 8
echo "✅ UR5e driver ready!"
