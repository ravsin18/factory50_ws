#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Factory 5.0 — Fresh Workspace Setup
# Runs automatically on devcontainer creation (postCreateCommand)
# Can also be run manually: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

WS=/workspaces/factory50_ws
SRC=$WS/src

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Factory 5.0 — Workspace Setup              ║"
echo "║   UR5e + Franka | MARL | Digital Twin        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Source ROS2 ────────────────────────────────────────────────────────────
source /opt/ros/jazzy/setup.bash
source /opt/franka_ws/install/setup.bash 2>/dev/null || true

# ── 2. Create src directory ───────────────────────────────────────────────────
mkdir -p $SRC && cd $SRC

# ── 3. Clone UR5e description ─────────────────────────────────────────────────
echo "[1/5] Fetching UR5e URDF..."
if [ ! -d "$SRC/Universal_Robots_ROS2_Description" ]; then
  git clone --depth 1 \
    https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git
fi

# ── 4. Clone Franka description ───────────────────────────────────────────────
echo "[2/5] Fetching Franka Panda URDF..."
if [ ! -d "$SRC/franka_description" ]; then
  git clone --depth 1 \
    https://github.com/frankaemika/franka_description.git
fi

# ── 5. Install rosdep dependencies ───────────────────────────────────────────
echo "[3/5] Installing ROS dependencies..."
cd $WS
rosdep update --rosdistro jazzy 2>/dev/null || true
rosdep install --from-paths src --ignore-src -r -y 2>/dev/null || true

# ── 6. Build workspace ────────────────────────────────────────────────────────
echo "[4/5] Building workspace..."
cd $WS
colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  2>&1 | tail -20

# ── 7. Source the built workspace ─────────────────────────────────────────────
echo "[5/5] Sourcing workspace..."
source $WS/install/setup.bash

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   ✅ Setup complete!                          ║"
echo "║                                              ║"
echo "║   Quick start:                               ║"
echo "║   ros2 launch factory50_simulation           ║"
echo "║         workcell.launch.py                   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
