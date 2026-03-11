"""
factory50_marl — MAPPO Agent
Multi-Agent PPO with Centralized Critic (CTDE architecture).

Designed to run on Google Colab (free T4/A100 GPU).
Export this file + dual_arm_packing_env.py to Colab and run train_mappo.ipynb.

Architecture:
  - 2 Actor networks (one per arm) — decentralized execution
  - 1 Centralized Critic — sees global state during training
  - GAE advantage estimation
  - PPO clipping with entropy bonus
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from typing import Dict, List, Tuple


# ── Network Architecture ───────────────────────────────────────────────────────
class ActorNetwork(nn.Module):
    """
    Per-agent actor: local_obs → action_mean + action_log_std
    Runs independently on each arm during execution (decentralized).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        # Log std as learnable parameter
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Orthogonal initialization (standard for PPO)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean    = self.net(obs)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def get_action(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        dist          = Normal(mean, log_std.exp())
        action        = dist.sample()
        log_prob      = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob, dist.entropy().sum(-1)

    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor):
        mean, log_std = self.forward(obs)
        dist          = Normal(mean, log_std.exp())
        log_prob      = dist.log_prob(action).sum(-1)
        entropy       = dist.entropy().sum(-1)
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Shared critic: global_obs → V(s)
    Sees BOTH agents' observations + human state during training.
    NOT used during execution (decentralized deployment).
    """

    def __init__(self, global_obs_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


# ── Rollout Buffer ─────────────────────────────────────────────────────────────
class RolloutBuffer:
    """Stores experience for one PPO update."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs_arm0    : List = []
        self.obs_arm1    : List = []
        self.obs_global  : List = []
        self.acts_arm0   : List = []
        self.acts_arm1   : List = []
        self.logp_arm0   : List = []
        self.logp_arm1   : List = []
        self.rewards     : List = []
        self.values      : List = []
        self.dones       : List = []

    def add(self, obs, actions, log_probs, reward, value, done):
        self.obs_arm0.append(obs['arm0'])
        self.obs_arm1.append(obs['arm1'])
        self.obs_global.append(obs['global'])
        self.acts_arm0.append(actions['arm0'])
        self.acts_arm1.append(actions['arm1'])
        self.logp_arm0.append(log_probs['arm0'])
        self.logp_arm1.append(log_probs['arm1'])
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def to_tensors(self, device):
        def t(x): return torch.FloatTensor(np.array(x)).to(device)
        return {
            'obs_arm0':   t(self.obs_arm0),
            'obs_arm1':   t(self.obs_arm1),
            'obs_global': t(self.obs_global),
            'acts_arm0':  t(self.acts_arm0),
            'acts_arm1':  t(self.acts_arm1),
            'logp_arm0':  t(self.logp_arm0),
            'logp_arm1':  t(self.logp_arm1),
            'rewards':    t(self.rewards),
            'values':     t(self.values),
            'dones':      t(self.dones),
        }


# ── MAPPO Trainer ──────────────────────────────────────────────────────────────
class MAPPOTrainer:
    """
    Multi-Agent PPO with Centralized Critic.

    Example (Colab):
        from dual_arm_packing_env import DualArmPackingEnv
        from mappo_agent import MAPPOTrainer

        env     = DualArmPackingEnv(use_human=True)
        trainer = MAPPOTrainer(env)
        trainer.train(total_steps=2_000_000, save_path='./models')
    """

    def __init__(self, env, config: Dict = None):
        self.env    = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training device: {self.device}")

        # ── Default Hyperparameters ────────────────────────────────────────────
        self.cfg = {
            'lr_actor':        3e-4,
            'lr_critic':       1e-3,
            'gamma':           0.99,
            'gae_lambda':      0.95,
            'clip_eps':        0.2,
            'entropy_coef':    0.01,
            'value_coef':      0.5,
            'max_grad_norm':   0.5,
            'rollout_steps':   4096,
            'n_epochs':        10,
            'batch_size':      256,
            'log_interval':    10,
        }
        if config:
            self.cfg.update(config)

        # ── Networks ──────────────────────────────────────────────────────────
        obs_sample, _ = env.reset()
        arm0_obs_dim  = obs_sample['arm0'].shape[0]    # 21
        arm1_obs_dim  = obs_sample['arm1'].shape[0]    # 20
        global_obs_dim= obs_sample['global'].shape[0]  # 48
        arm0_act_dim  = env.action_space['arm0'].shape[0]   # 7
        arm1_act_dim  = env.action_space['arm1'].shape[0]   # 8

        self.actor_arm0 = ActorNetwork(arm0_obs_dim,  arm0_act_dim).to(self.device)
        self.actor_arm1 = ActorNetwork(arm1_obs_dim,  arm1_act_dim).to(self.device)
        self.critic     = CentralizedCritic(global_obs_dim).to(self.device)

        self.opt_actor0 = optim.Adam(self.actor_arm0.parameters(), lr=self.cfg['lr_actor'])
        self.opt_actor1 = optim.Adam(self.actor_arm1.parameters(), lr=self.cfg['lr_actor'])
        self.opt_critic = optim.Adam(self.critic.parameters(),     lr=self.cfg['lr_critic'])

        self.buffer     = RolloutBuffer()
        self.ep_rewards : List[float] = []

    def train(self, total_steps: int, save_path: str = './models'):
        """Main training loop. Run on Colab for best performance."""
        import os
        os.makedirs(save_path, exist_ok=True)

        obs, _     = self.env.reset()
        step       = 0
        episode    = 0
        ep_reward  = 0.0
        best_reward= -np.inf

        print(f"\nStarting MAPPO training | Total steps: {total_steps:,}")
        print(f"Rollout: {self.cfg['rollout_steps']} | Batch: {self.cfg['batch_size']}")
        print("-" * 60)

        while step < total_steps:

            # ── Collect Rollout ────────────────────────────────────────────────
            self.buffer.reset()

            for _ in range(self.cfg['rollout_steps']):
                with torch.no_grad():
                    obs0_t = torch.FloatTensor(obs['arm0']).unsqueeze(0).to(self.device)
                    obs1_t = torch.FloatTensor(obs['arm1']).unsqueeze(0).to(self.device)
                    obs_g  = torch.FloatTensor(obs['global']).unsqueeze(0).to(self.device)

                    act0, lp0, _ = self.actor_arm0.get_action(obs0_t)
                    act1, lp1, _ = self.actor_arm1.get_action(obs1_t)
                    value        = self.critic(obs_g)

                actions = {
                    'arm0': act0.squeeze(0).cpu().numpy(),
                    'arm1': act1.squeeze(0).cpu().numpy(),
                }
                log_probs = {
                    'arm0': lp0.item(),
                    'arm1': lp1.item(),
                }

                next_obs, reward, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated

                self.buffer.add(obs, actions, log_probs, reward, value.item(), done)

                obs        = next_obs
                ep_reward += reward
                step      += 1

                if done:
                    self.ep_rewards.append(ep_reward)
                    episode   += 1
                    ep_reward  = 0.0
                    obs, _     = self.env.reset()

            # ── PPO Update ────────────────────────────────────────────────────
            loss_info = self._ppo_update()

            # ── Logging ───────────────────────────────────────────────────────
            if episode % self.cfg['log_interval'] == 0 and self.ep_rewards:
                mean_r = np.mean(self.ep_rewards[-50:])
                print(f"Step {step:8,} | Ep {episode:5d} | "
                      f"Mean reward (50ep): {mean_r:8.2f} | "
                      f"Actor loss: {loss_info['actor_loss']:.4f} | "
                      f"Critic loss: {loss_info['critic_loss']:.4f}")

                # Save best model
                if mean_r > best_reward:
                    best_reward = mean_r
                    self.save(f"{save_path}/best_model.pt")
                    print(f"  → New best: {best_reward:.2f} ✓")

            # Save checkpoint every 100k steps
            if step % 100_000 == 0:
                self.save(f"{save_path}/checkpoint_{step}.pt")

        self.save(f"{save_path}/final_model.pt")
        print(f"\nTraining complete | Best reward: {best_reward:.2f}")

    def _compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_gae   = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1] * (1 - dones[t + 1])

            delta       = rewards[t] + self.cfg['gamma'] * next_val - values[t]
            last_gae    = delta + self.cfg['gamma'] * self.cfg['gae_lambda'] * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def _ppo_update(self) -> Dict:
        """Run PPO update for all agents."""
        data   = self.buffer.to_tensors(self.device)
        adv, ret = self._compute_gae(
            data['rewards'].cpu().numpy(),
            data['values'].cpu().numpy(),
            data['dones'].cpu().numpy()
        )
        adv_t = torch.FloatTensor(adv).to(self.device)
        ret_t = torch.FloatTensor(ret).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)  # normalize

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        n_steps = len(data['rewards'])

        for _ in range(self.cfg['n_epochs']):
            idx = np.random.permutation(n_steps)

            for start in range(0, n_steps, self.cfg['batch_size']):
                b = idx[start:start + self.cfg['batch_size']]
                bt = torch.LongTensor(b).to(self.device)

                # ── Actor 0 (UR5e) update ──────────────────────────────────────
                new_lp0, ent0 = self.actor_arm0.evaluate_action(
                    data['obs_arm0'][bt], data['acts_arm0'][bt])
                ratio0    = (new_lp0 - data['logp_arm0'][bt]).exp()
                clip0     = torch.clamp(ratio0, 1-self.cfg['clip_eps'], 1+self.cfg['clip_eps'])
                aloss0    = -torch.min(ratio0 * adv_t[bt], clip0 * adv_t[bt]).mean()
                aloss0   -= self.cfg['entropy_coef'] * ent0.mean()

                self.opt_actor0.zero_grad()
                aloss0.backward()
                nn.utils.clip_grad_norm_(self.actor_arm0.parameters(), self.cfg['max_grad_norm'])
                self.opt_actor0.step()

                # ── Actor 1 (Franka) update ────────────────────────────────────
                new_lp1, ent1 = self.actor_arm1.evaluate_action(
                    data['obs_arm1'][bt], data['acts_arm1'][bt])
                ratio1    = (new_lp1 - data['logp_arm1'][bt]).exp()
                clip1     = torch.clamp(ratio1, 1-self.cfg['clip_eps'], 1+self.cfg['clip_eps'])
                aloss1    = -torch.min(ratio1 * adv_t[bt], clip1 * adv_t[bt]).mean()
                aloss1   -= self.cfg['entropy_coef'] * ent1.mean()

                self.opt_actor1.zero_grad()
                aloss1.backward()
                nn.utils.clip_grad_norm_(self.actor_arm1.parameters(), self.cfg['max_grad_norm'])
                self.opt_actor1.step()

                # ── Centralized Critic update ──────────────────────────────────
                values_pred = self.critic(data['obs_global'][bt])
                vloss       = nn.functional.mse_loss(values_pred, ret_t[bt])
                vloss      *= self.cfg['value_coef']

                self.opt_critic.zero_grad()
                vloss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg['max_grad_norm'])
                self.opt_critic.step()

                total_actor_loss  += (aloss0.item() + aloss1.item()) / 2
                total_critic_loss += vloss.item()

        n_updates = self.cfg['n_epochs'] * (n_steps // self.cfg['batch_size'])
        return {
            'actor_loss':  total_actor_loss  / n_updates,
            'critic_loss': total_critic_loss / n_updates,
        }

    def save(self, path: str):
        torch.save({
            'actor_arm0': self.actor_arm0.state_dict(),
            'actor_arm1': self.actor_arm1.state_dict(),
            'critic':     self.critic.state_dict(),
            'config':     self.cfg,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor_arm0.load_state_dict(ckpt['actor_arm0'])
        self.actor_arm1.load_state_dict(ckpt['actor_arm1'])
        self.critic.load_state_dict(ckpt['critic'])
        print(f"Loaded model from {path}")
