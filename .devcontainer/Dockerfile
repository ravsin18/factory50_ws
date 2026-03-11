FROM osrf/ros:jazzy-desktop

# ─────────────────────────────────────────────────────────────────────────────
# Factory 5.0 — Dockerfile (FIXED for ROS2 Jazzy)
# Key fix: Jazzy uses gz_*_vendor packages, NOT ros-jazzy-gazebo-ros-pkgs
# GPU: GTX 1650 Ti (4GB VRAM) | WSL2 + Docker Desktop
# ─────────────────────────────────────────────────────────────────────────────

ENV DEBIAN_FRONTEND=noninteractive

# ── 1. System Tools ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    nano \
    htop \
    python3-opencv \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Gazebo via VENDOR packages (correct Jazzy method) ──────────────────────
# OLD (wrong): ros-jazzy-gazebo-ros-pkgs, ros-jazzy-ros-gz-bridge
# NEW (correct): ros-jazzy-gz-*-vendor + ros-jazzy-ros-gz-*
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-gz-sim-vendor \
    ros-jazzy-gz-tools-vendor \
    ros-jazzy-gz-physics-vendor \
    ros-jazzy-gz-sensors-vendor \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-interfaces \
    && rm -rf /var/lib/apt/lists/*

# ── 3. Robot Description & Visualization ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-xacro \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-tf-transformations \
    ros-jazzy-tf2-tools \
    ros-jazzy-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# ── 4. MoveIt2 ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-moveit \
    ros-jazzy-moveit-ros-planning-interface \
    ros-jazzy-moveit-visual-tools \
    && rm -rf /var/lib/apt/lists/*

# ── 5. ros2_control + Gazebo integration ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-gz-ros2-control \
    && rm -rf /var/lib/apt/lists/*

# ── 6. UR5e Driver ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ur \
    ros-jazzy-ur-moveit-config \
    && rm -rf /var/lib/apt/lists/*

# ── 7. Foxglove Bridge ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# ── 8. PyTorch CPU (training on Colab — keeps image lean) ────────────────────
RUN pip3 install --break-system-packages --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# ── 9. RL + Perception + Digital Twin + Utilities ────────────────────────────
RUN pip3 install --break-system-packages --no-cache-dir \
    gymnasium==0.29.1 \
    stable-baselines3 \
    tensorboard \
    numpy \
    scipy \
    mediapipe \
    opencv-python-headless \
    asyncua \
    influxdb-client \
    aiohttp \
    matplotlib \
    pandas \
    pytest \
    black \
    ipython

# ── 10. Environment & Shell ───────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=dialog
ENV ROS_DOMAIN_ID=42
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /workspaces/factory50_ws/install/setup.bash ]; then" >> /root/.bashrc && \
    echo "  source /workspaces/factory50_ws/install/setup.bash" >> /root/.bashrc && \
    echo "fi" >> /root/.bashrc

WORKDIR /workspaces/factory50_ws
