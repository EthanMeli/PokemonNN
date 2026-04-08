"""
rollout_buffer.py
=================
Collects transitions from completed self-play battles and computes GAE
advantages + returns ready for PPO minibatch updates.

Key design choices:
    - Each Pokemon battle is a *complete* episode by the time it lands here,
      the trajectory always ends with done=True. There is no bootstrap value
      at the end of an episode (V_terminal = 0).
    
    - We compute GAE per-trajectory and concatenate, rather than treating
      the whole buffer as one long sequence (battles are independent)

    - Observations are stored as CPU tensors and stacked into batched tensors
      only at the moment we hand them to the trainer, to keep memory bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

@dataclass
class RolloutBatch:
    """Flat, batches tensors ready for the PPO update loop."""
    species_indices: torch.Tensor   # (N, 12)
    move_indices: torch.Tensor      # (N, 12, 4)
    numeric_features: torch.Tensor  # (N, 12, 162)
    context_features: torch.Tensor  # (N, 24)
    action_masks: torch.Tensor      # (N, 9) bool
    actions: torch.Tensor           # (N,) long
    old_log_probs: torch.Tensor     # (N,) float
    values: torch.Tensor            # (N,) float
    advantages: torch.Tensor        # (N,) float
    returns: torch.Tensor           # (N,) float

    def __len__(self) -> int:
        return self.actions.shape[0]
    
    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            species_indices=self.species_indices.to(device),
            move_indices=self.move_indices.to(device),
            numeric_features=self.numeric_features.to(device),
            context_features=self.context_features.to(device),
            action_masks=self.action_masks.to(device),
            actions=self.actions.to(device),
            old_log_probs=self.old_log_probs.to(device),
            values=self.values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )
    
def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
    """
    Generalized Advantage Estimation (Schulman et al. 2015)

    For an episode that ends at step T (dones[T-1] = True):
        delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
        A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
    
    Since each battle is a complete episode here, V_{T} (after the last 
    transition) is 0 and the recursion terminates cleanly.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0 # episode complete -> no bootstrap
            next_non_terminal = 0.0
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    return advantages
    
class RolloutBuffer:
    """
    Accumulates trajectories from self-play and produces a flat RolloutBatch.

    Usage:
        buf = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        for traj in agent.pop_trajectories():
            buf.add_trajectory(traj)
        batch = buf.build_batch()
        buf.reset()
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95,
                 normalize_advantages: bool = True):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.reset()

    def reset(self):
        self._obs_species: List[torch.Tensor] = []
        self._obs_moves: List[torch.Tensor] = []
        self._obs_numeric: List[torch.Tensor] = []
        self._obs_context: List[torch.Tensor] = []
        self._obs_mask: List[torch.Tensor] = []
        self._actions: List[int] = []
        self._log_probs: List[float] = []
        self._values: List[float] = []
        self._advantages: List[np.ndarray] = []
        self._returns: List[np.ndarray] = []
        self._size = 0

    def __len__(self) -> int:
        return self._size
    
    def add_trajectory(self, trajectory: List) -> None:
        """Append one full battle trajectory and compute its GAE."""
        if not trajectory:
            return
        
        rewards = np.array([t.reward for t in trajectory], dtype=np.float32)
        values = np.array([t.value for t in trajectory], dtype=np.float32)
        dones = np.array([t.done for t in trajectory], dtype=np.float32)

        advantages = compute_gae(rewards, values, dones,
                                 self.gamma, self.gae_lambda)
        returns = advantages + values

        self._advantages.append(advantages)
        self._returns.append(returns)

        for t in trajectory:
            # obs entries already have a leading batch dim of 1 -> squeeze it
            self._obs_species.append(t.obs["species_indices"].squeeze(0))
            self._obs_moves.append(t.obs["move_indices"].squeeze(0))
            self._obs_numeric.append(t.obs["numeric_features"].squeeze(0))
            self._obs_context.append(t.obs["context_features"].squeeze(0))
            self._obs_mask.append(t.obs["action_mask"].squeeze(0))
            self._actions.append(t.action)
            self._log_probs.append(t.log_prob)
            self._values.append(t.value)
            self._size += 1

    def build_batch(self) -> RolloutBatch:
        """Stack everything into a flat RolloutBatch."""
        advantages = np.concatenate(self._advantages) if self._advantages else np.zeros(0, dtype=np.float32)
        returns = np.concatenate(self._returns) if self._returns else np.zeros(0, dtype=np.float32)

        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return RolloutBatch(
            species_indices=torch.stack(self._obs_species).long(),
            move_indices=torch.stack(self._obs_moves).long(),
            numeric_features=torch.stack(self._obs_numeric).float(),
            context_features=torch.stack(self._obs_context).float(),
            action_masks=torch.stack(self._obs_mask).bool(),
            actions=torch.tensor(self._actions, dtype=torch.long),
            old_log_probs=torch.tensor(self._log_probs, dtype=torch.float32),
            values=torch.tensor(self._values, dtype=torch.float32),
            advantages=torch.from_numpy(advantages),
            returns=torch.from_numpy(returns),
        )