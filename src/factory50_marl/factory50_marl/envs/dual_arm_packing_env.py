"""
factory50_marl — DualArmPackingEnv
Multi-Agent Gymnasium environment for the Factory 5.0 dual-arm box packing cell.

Architecture: CTDE (Centralized Training, Decentralized Execution)
  - Agent 0: UR5e  — picks boxes from input conveyor, places on transfer fixture
  - Agent 1: Franka Panda — picks from fixture, packs into output pallet

Observation spaces (per agent):
  - Local obs: joint angles, EEF pose, gripper state, task progress
  - Global obs (training only): both agents + human state → fed to central critic

Action space: continuous joint velocity deltas + gripper command

Compatible with: Google Colab training (export env, train there, import model back)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional


# ── Environment Constants ──────────────────────────────────────────────────────
N_AGENTS        = 2
UR5E_DOF        = 6
FRANKA_DOF      = 7
MAX_STEPS       = 500
DT              = 0.05          # 20 Hz control loop

# Joint limits (simplified — replace with real URDF limits)
UR5E_JOINT_LIMITS   = np.array([[-2*np.pi]*6,  [2*np.pi]*6])
FRANKA_JOINT_LIMITS = np.array([[-2.7, -1.7, -2.7, -3.0, -2.7, 0.0, -2.7],
                                  [ 2.7,  1.7,  2.7, -0.1,  2.7, 3.7,  2.7]])

# Workspace positions (metres, in world frame)
INPUT_BIN_POS   = np.array([-0.6,  0.4, 0.1])   # UR5e picks from here
FIXTURE_POS     = np.array([ 0.0,  0.5, 0.2])   # handover fixture
PALLET_POS      = np.array([ 0.6,  0.4, 0.1])   # Franka packs here

# Safety thresholds
COLLISION_DIST  = 0.08    # metres — arm-to-arm collision threshold
HUMAN_STOP_DIST = 0.5     # metres — red zone


class DualArmPackingEnv(gym.Env):
    """
    Dual-arm box packing environment for MAPPO training.

    Usage (local testing):
        env = DualArmPackingEnv()
        obs, info = env.reset()
        obs_arm0, obs_arm1 = obs['arm0'], obs['arm1']

    Usage (Colab training):
        Export this file to Colab, wrap with MARLWrapper, train with MAPPO.
        See: notebooks/train_mappo.ipynb
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None, use_human=True):
        super().__init__()
        self.render_mode = render_mode
        self.use_human   = use_human  # False during curriculum stage 1-2
        self.step_count  = 0

        # ── Per-Agent Observation Spaces ──────────────────────────────────────
        # ARM 0 (UR5e): joints(6) + EEF(6) + gripper(1) + box_pos(3) +
        #               fixture_state(1) + human_dist(1) + arm1_eef(3) = 21
        arm0_obs_dim = UR5E_DOF + 6 + 1 + 3 + 1 + 1 + 3   # = 21

        # ARM 1 (Franka): joints(7) + EEF(6) + gripper(1) + fixture_state(1) +
        #                 box_in_fixture(1) + human_dist(1) + arm0_eef(3) = 20
        arm1_obs_dim = FRANKA_DOF + 6 + 1 + 1 + 1 + 1 + 3  # = 20

        self.observation_space = spaces.Dict({
            'arm0': spaces.Box(-np.inf, np.inf, (arm0_obs_dim,), np.float32),
            'arm1': spaces.Box(-np.inf, np.inf, (arm1_obs_dim,), np.float32),
            # Global obs for centralized critic (training only)
            'global': spaces.Box(-np.inf, np.inf, (arm0_obs_dim + arm1_obs_dim + 4 + 3,), np.float32)
        })

        # ── Per-Agent Action Spaces ────────────────────────────────────────────
        # ARM 0 (UR5e):   joint vel deltas(6) + gripper(1) = 7
        # ARM 1 (Franka): joint vel deltas(7) + gripper(1) = 8
        self.action_space = spaces.Dict({
            'arm0': spaces.Box(-1.0, 1.0, (UR5E_DOF + 1,),   np.float32),
            'arm1': spaces.Box(-1.0, 1.0, (FRANKA_DOF + 1,), np.float32),
        })

        # ── Internal State ─────────────────────────────────────────────────────
        self._init_state()

    def _init_state(self):
        """Initialise all simulation state variables."""
        # Joint angles (random init within limits)
        self.arm0_joints  = np.zeros(UR5E_DOF,   dtype=np.float32)
        self.arm1_joints  = np.zeros(FRANKA_DOF, dtype=np.float32)

        # End-effector positions (world frame, simplified FK)
        self.arm0_eef     = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        self.arm1_eef     = np.array([ 0.5, 0.0, 0.5], dtype=np.float32)

        # Gripper states (0=open, 1=closed)
        self.arm0_gripper = 0.0
        self.arm1_gripper = 0.0

        # Box state
        self.box_pos          = INPUT_BIN_POS.copy().astype(np.float32)
        self.box_held_by      = None   # None / 'arm0' / 'arm1'
        self.box_in_fixture   = False
        self.box_packed       = False

        # Human state
        self.human_pos        = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        self.human_vel        = np.zeros(3, dtype=np.float32)

        # Task phase: 0=pick_box, 1=place_fixture, 2=handover, 3=pack
        self.task_phase       = 0
        self.step_count       = 0
        self.handover_success = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()

        # Add small random noise to initial joint angles
        self.arm0_joints += self.np_random.uniform(-0.1, 0.1, UR5E_DOF).astype(np.float32)
        self.arm1_joints += self.np_random.uniform(-0.1, 0.1, FRANKA_DOF).astype(np.float32)

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, actions: Dict) -> Tuple:
        self.step_count += 1

        # ── Apply Actions ─────────────────────────────────────────────────────
        act0 = np.clip(actions['arm0'], -1.0, 1.0)
        act1 = np.clip(actions['arm1'], -1.0, 1.0)

        # Scale joint velocity deltas
        self.arm0_joints  = np.clip(
            self.arm0_joints + act0[:UR5E_DOF]   * 0.05,
            UR5E_JOINT_LIMITS[0], UR5E_JOINT_LIMITS[1]
        ).astype(np.float32)
        self.arm1_joints  = np.clip(
            self.arm1_joints + act1[:FRANKA_DOF] * 0.05,
            FRANKA_JOINT_LIMITS[0], FRANKA_JOINT_LIMITS[1]
        ).astype(np.float32)

        # Gripper commands
        self.arm0_gripper = float(act0[-1] > 0.0)
        self.arm1_gripper = float(act1[-1] > 0.0)

        # Update simplified EEF positions (FK approximation — replace with MoveIt2 FK)
        self._update_eef()

        # Update human position (random walk when use_human=True)
        if self.use_human:
            self._update_human()

        # Update task logic
        self._update_task_logic()

        # Compute rewards
        rewards = self._compute_rewards()

        # Check termination
        terminated = self.box_packed
        truncated  = self.step_count >= MAX_STEPS

        obs  = self._get_obs()
        info = self._get_info()
        info['rewards'] = rewards

        # Total reward = local + global team reward
        total_reward = rewards['arm0'] + rewards['arm1'] + rewards['team']

        return obs, total_reward, terminated, truncated, info

    def _update_eef(self):
        """
        Simplified EEF position from joint angles.
        Replace with proper FK using robot_description + kdl_parser in ROS2 integration.
        """
        # UR5e: very rough approximation of reach
        self.arm0_eef = np.array([
            np.sin(self.arm0_joints[0]) * 0.8,
            np.cos(self.arm0_joints[0]) * 0.8,
            0.3 + np.sin(self.arm0_joints[2]) * 0.4
        ], dtype=np.float32)

        # Franka: offset to right side of cell
        self.arm1_eef = np.array([
            0.4 + np.sin(self.arm1_joints[0]) * 0.6,
            np.cos(self.arm1_joints[0]) * 0.6,
            0.3 + np.sin(self.arm1_joints[2]) * 0.4
        ], dtype=np.float32)

        # If arm is holding box, move box with it
        if self.box_held_by == 'arm0':
            self.box_pos = self.arm0_eef.copy()
        elif self.box_held_by == 'arm1':
            self.box_pos = self.arm1_eef.copy()

    def _update_human(self):
        """Simulate human random walk toward/away from workcell."""
        noise = self.np_random.uniform(-0.02, 0.02, 3).astype(np.float32)
        # Bias toward workcell centre occasionally
        if self.np_random.random() < 0.01:
            noise[1] -= 0.1   # walk closer
        elif self.np_random.random() < 0.01:
            noise[1] += 0.1   # walk away

        self.human_pos = np.clip(
            self.human_pos + noise,
            [-1.5, 0.3, 0.0], [1.5, 3.0, 0.0]
        ).astype(np.float32)

    def _update_task_logic(self):
        """State machine: track task phase transitions."""
        # Phase 0: UR5e picks box from input bin
        if self.task_phase == 0:
            dist_to_box = np.linalg.norm(self.arm0_eef - self.box_pos)
            if dist_to_box < 0.12 and self.arm0_gripper > 0.5:
                self.box_held_by  = 'arm0'
                self.task_phase   = 1

        # Phase 1: UR5e places box on fixture
        elif self.task_phase == 1:
            dist_to_fix = np.linalg.norm(self.arm0_eef - FIXTURE_POS)
            if dist_to_fix < 0.1 and self.arm0_gripper < 0.5 and self.box_held_by == 'arm0':
                self.box_held_by    = None
                self.box_pos        = FIXTURE_POS.copy().astype(np.float32)
                self.box_in_fixture = True
                self.task_phase     = 2

        # Phase 2: Handover — wait for UR5e to retract, Franka approaches
        elif self.task_phase == 2:
            arm0_retracted = np.linalg.norm(self.arm0_eef - FIXTURE_POS) > 0.3
            arm1_at_fix    = np.linalg.norm(self.arm1_eef - FIXTURE_POS) < 0.12
            if arm0_retracted and arm1_at_fix and self.arm1_gripper > 0.5:
                self.box_held_by      = 'arm1'
                self.box_in_fixture   = False
                self.handover_success = True
                self.task_phase       = 3

        # Phase 3: Franka packs box into pallet
        elif self.task_phase == 3:
            dist_to_pal = np.linalg.norm(self.arm1_eef - PALLET_POS)
            if dist_to_pal < 0.1 and self.arm1_gripper < 0.5 and self.box_held_by == 'arm1':
                self.box_held_by  = None
                self.box_pos      = PALLET_POS.copy().astype(np.float32)
                self.box_packed   = True

    def _compute_rewards(self) -> Dict:
        rewards = {'arm0': 0.0, 'arm1': 0.0, 'team': 0.0}

        # ── Time penalty (both agents) ────────────────────────────────────────
        rewards['arm0'] -= 0.01
        rewards['arm1'] -= 0.01

        # ── Arm 0 (UR5e) rewards ──────────────────────────────────────────────
        if self.task_phase == 0:
            d = np.linalg.norm(self.arm0_eef - self.box_pos)
            rewards['arm0'] += max(0, 0.5 - d) * 0.1     # approach box

        elif self.task_phase == 1:
            d = np.linalg.norm(self.arm0_eef - FIXTURE_POS)
            rewards['arm0'] += max(0, 0.5 - d) * 0.1     # approach fixture
            if self.box_in_fixture:
                rewards['arm0'] += 5.0                    # placed on fixture ✓

        # ── Arm 1 (Franka) rewards ────────────────────────────────────────────
        if self.task_phase == 2 and self.box_in_fixture:
            d = np.linalg.norm(self.arm1_eef - FIXTURE_POS)
            rewards['arm1'] += max(0, 0.5 - d) * 0.1     # approach fixture

        elif self.task_phase == 3:
            d = np.linalg.norm(self.arm1_eef - PALLET_POS)
            rewards['arm1'] += max(0, 0.5 - d) * 0.1     # approach pallet
            if self.box_packed:
                rewards['arm1'] += 10.0                   # box packed ✓

        # ── Team Rewards & Penalties ──────────────────────────────────────────
        if self.box_packed:
            rewards['team'] += 20.0                       # 🏆 full cycle complete

        if self.handover_success:
            rewards['team'] += 5.0
            self.handover_success = False                 # only reward once

        # Inter-arm collision penalty
        arm_dist = np.linalg.norm(self.arm0_eef - self.arm1_eef)
        if arm_dist < COLLISION_DIST:
            rewards['team'] -= 15.0

        # Human safety penalties (if human present)
        if self.use_human:
            h0 = np.linalg.norm(self.arm0_eef - self.human_pos)
            h1 = np.linalg.norm(self.arm1_eef - self.human_pos)
            if h0 < HUMAN_STOP_DIST or h1 < HUMAN_STOP_DIST:
                rewards['team'] -= 10.0                   # safety violation!

        return rewards

    def _get_obs(self) -> Dict:
        human_dist_arm0 = float(np.linalg.norm(self.arm0_eef - self.human_pos))
        human_dist_arm1 = float(np.linalg.norm(self.arm1_eef - self.human_pos))

        obs_arm0 = np.concatenate([
            self.arm0_joints,                             # 6
            self.arm0_eef,                                # 3
            np.array([0.0, 0.0, 0.0]),                   # EEF orientation (simplified)
            np.array([self.arm0_gripper]),                # 1
            self.box_pos,                                 # 3
            np.array([float(self.box_in_fixture)]),       # 1
            np.array([human_dist_arm0]),                  # 1
            self.arm1_eef,                                # 3  (partner awareness)
        ]).astype(np.float32)

        obs_arm1 = np.concatenate([
            self.arm1_joints,                             # 7
            self.arm1_eef,                                # 3
            np.array([0.0, 0.0, 0.0]),                   # EEF orientation (simplified)
            np.array([self.arm1_gripper]),                # 1
            np.array([float(self.box_in_fixture)]),       # 1
            np.array([float(self.box_held_by == 'arm1')]),# 1
            np.array([human_dist_arm1]),                  # 1
            self.arm0_eef,                                # 3  (partner awareness)
        ]).astype(np.float32)

        # Global observation for centralized critic
        task_phase_onehot = np.eye(4)[self.task_phase].astype(np.float32)
        obs_global = np.concatenate([
            obs_arm0,
            obs_arm1,
            self.human_pos,
            task_phase_onehot
        ]).astype(np.float32)

        return {'arm0': obs_arm0, 'arm1': obs_arm1, 'global': obs_global}

    def _get_info(self) -> Dict:
        return {
            'task_phase':       self.task_phase,
            'box_in_fixture':   self.box_in_fixture,
            'box_packed':       self.box_packed,
            'step':             self.step_count,
            'arm_distance':     float(np.linalg.norm(self.arm0_eef - self.arm1_eef)),
            'human_dist_arm0':  float(np.linalg.norm(self.arm0_eef - self.human_pos)),
            'human_dist_arm1':  float(np.linalg.norm(self.arm1_eef - self.human_pos)),
        }

    def render(self):
        if self.render_mode == 'human':
            info = self._get_info()
            phase_names = ['Pick Box', 'Place Fixture', 'Handover', 'Pack Pallet']
            print(f"Step {self.step_count:4d} | "
                  f"Phase: {phase_names[self.task_phase]:15s} | "
                  f"Arm dist: {info['arm_distance']:.2f}m | "
                  f"Human: {info['human_dist_arm0']:.2f}m/{info['human_dist_arm1']:.2f}m")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing DualArmPackingEnv...")
    env = DualArmPackingEnv(render_mode='human', use_human=True)

    obs, info = env.reset()
    print(f"Arm0 obs shape: {obs['arm0'].shape}")   # (21,)
    print(f"Arm1 obs shape: {obs['arm1'].shape}")   # (20,)
    print(f"Global obs shape: {obs['global'].shape}")

    total_reward = 0
    for step in range(50):
        actions = {
            'arm0': env.action_space['arm0'].sample(),
            'arm1': env.action_space['arm1'].sample(),
        }
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        env.render()
        if terminated or truncated:
            break

    print(f"\nTotal reward over {step+1} steps: {total_reward:.2f}")
    print("✅ Environment test passed!")
    env.close()
