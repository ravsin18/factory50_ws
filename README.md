# 🏭 Factory 5.0 — Dual-Arm MARL Workcell

> **Multi-Agent Reinforcement Learning for Human-Centric Industrial Robotics**
> UR5e + Franka Panda | CTDE MAPPO | ISO 10218 HRC Safety | Real-Time Digital Twin

[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue)](https://docs.ros.org/en/jazzy/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Overview

A fully simulated **Factory 5.0** robotic workcell implementing:

| Pillar | Implementation |
|--------|---------------|
| 🤝 **Human-Robot Collaboration** | MediaPipe skeleton detection + ISO 10218 safety zones |
| 🧠 **Multi-Agent RL** | MAPPO (CTDE) — UR5e + Franka Panda box packing pipeline |
| 🔁 **Digital Twin** | OPC-UA server + real-time Grafana KPI dashboard |

**Simulation stack**: ROS2 Jazzy + Gazebo Harmonic + MoveIt2
**Training**: Google Colab (T4 GPU) → deploy model locally

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    factory50_ws                          │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │  UR5e (Arm0) │    │Franka (Arm1) │  ← Gazebo Sim   │
│  │  Pick box    │───▶│  Pack pallet │                  │
│  └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                           │
│  ┌──────▼───────────────────▼───────┐                  │
│  │     MARL Coordinator Node         │                  │
│  │  Handover protocol + Task FSM     │                  │
│  └──────────────────┬────────────────┘                  │
│                     │                                   │
│  ┌──────────────────▼────────────────┐                  │
│  │       HRC Safety Node             │  ← MediaPipe    │
│  │  Human pose → ISO 10218 zones     │                  │
│  └──────────────────┬────────────────┘                  │
│                     │                                   │
│  ┌──────────────────▼────────────────┐                  │
│  │     Digital Twin Bridge           │                  │
│  │  OPC-UA ──▶ InfluxDB ──▶ Grafana │                  │
│  └───────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Windows 11 + WSL2 + Docker Desktop
- NVIDIA GPU (GTX 1650 Ti or better)
- VS Code + Dev Containers extension

### 1. Clone & Open in Dev Container
```bash
git clone https://github.com/YOUR_USERNAME/factory50_ws.git
cd factory50_ws
code .
# VS Code: "Reopen in Container" → builds Docker image automatically
```

### 2. Start Companion Services (new terminal)
```bash
docker-compose up -d
# Grafana: http://localhost:3000  (admin / factory50)
# InfluxDB: http://localhost:8086
```

### 3. Build the Workspace
```bash
# Inside devcontainer terminal:
colcon build --symlink-install
source install/setup.bash
```

### 4. Launch the Full Workcell
```bash
ros2 launch factory50_simulation workcell.launch.py
```

### 5. Train on Google Colab
```
notebooks/Factory50_MAPPO_Training.ipynb
→ Upload env + agent files → Run all cells → Download model
→ Place model in: src/factory50_marl/models/best_model.pt
```

---

## 📦 Package Structure

```
src/
├── factory50_description/    # UR5e + Franka URDFs, meshes
├── factory50_simulation/     # Gazebo world, launch files
├── factory50_perception/     # HRC safety node (MediaPipe)
├── factory50_control/        # MoveIt2 + velocity scaling
├── factory50_marl/           # MAPPO training + Gym env
├── factory50_coordinator/    # Handover protocol + task FSM
└── factory50_twin/           # OPC-UA bridge + Grafana
```

---

## 🤖 MARL Architecture (CTDE)

```
Training (Centralized):              Execution (Decentralized):
┌──────────────────────┐            ┌──────────┐  ┌──────────┐
│   Central Critic     │            │ Actor 0  │  │ Actor 1  │
│  global_obs (48-dim) │            │  (UR5e)  │  │(Franka)  │
│  → V(s)              │            │ 21-dim   │  │ 20-dim   │
└──────────────────────┘            │ obs→act  │  │ obs→act  │
                                    └──────────┘  └──────────┘
                                       100 Hz        100 Hz
                                    (independent, no comms)
```

**Algorithm**: MAPPO | **Framework**: PyTorch | **Training**: Colab T4

---

## 🛡️ HRC Safety Zones (ISO 10218)

| Zone | Distance | Robot Behaviour |
|------|----------|-----------------|
| 🟢 Green  | > 1.2 m | Full speed (100%) |
| 🟡 Yellow | 0.5–1.2 m | Reduced speed (40%) |
| 🔴 Red    | < 0.5 m  | Immediate stop + retract |

---

## 📊 Digital Twin

Real-time data streams to **Grafana dashboard**:

- Joint states & EEF positions (100 Hz → OPC-UA)
- Safety zone transitions & human proximity (30 Hz)
- RL episode rewards & task phase (10 Hz)
- Production KPIs: cycle time, boxes packed/hour (1 Hz)

Connect **Foxglove Studio** to `ws://localhost:8765` for 3D visualization.

---

## 🇩🇪 German Industry Relevance

| Company | Relevant Module |
|---------|----------------|
| KUKA (Augsburg) | ROS2 + MoveIt2 + Digital Twin |
| Neura Robotics (Metzingen) | HRC + MARL + Safety |
| Siemens Digital Industries | OPC-UA + Grafana + Industry 4.0 |
| GEBHARDT Intralogistics | Box packing pipeline + ROS2 |
| BMW / Mercedes R&D | RL + sim-to-real + digital twin |

---

## 🛠️ Tech Stack

`ROS2 Jazzy` `Gazebo Harmonic` `MoveIt2` `PyTorch` `MediaPipe`
`MAPPO` `Gymnasium` `OPC-UA` `InfluxDB` `Grafana` `Foxglove`
`Docker` `WSL2` `Python 3.10` `C++17`

---

## 📄 License
MIT License — see [LICENSE](LICENSE)

---

*MSc Mechatronics — Factory 5.0 Portfolio Project | Germany*
