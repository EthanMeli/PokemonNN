"""
ppo_trainer.py
==============
PPO update logic for the Gen 1 Pokemon Agent.

Implements:
    - Clipped surrogate policy loss
    - Clipped value loss
    - Entropy bonus
    - Gradient clipping
    - Linear LR annealing across the full training run
    - Minibatch SGD over the rollout buffer (multiple epochs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .rollout_buffer import RolloutBatch


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO update loop"""
    learning_rate: float = 3e-4
    num_epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2            # PPO clip epsilon
    vf_clip_coef: float = 0.2         # value function clip
    vf_coef: float = 0.5              # value loss weight
    ent_coef: float = 0.01            # entropy bonus weight
    max_grad_norm: float = 0.5 
    target_kl: Optional[float] = 0.03 # early stop if mean KL exceeds this
    anneal_lr: bool = True

class PPOTrainer:
    """
    PPO updater. Owns the optimizer and LR schedule; the model itself lives
    outside (so it can be shared across rollout workers and PPO updates).
    """

    def __init__(self, model: nn.Module, config: PPOConfig,
                total_updates: int, device: torch.device):
        self.model = model
        self.config = config
        self.total_updates = total_updates
        self.device = device
        self._update_idx = 0

        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)

    # LR Schedule
    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _anneal_lr(self) -> float:
        """Linear annealing from config.learning_rate -> 0 across total_updates"""
        if not self.config.anneal_lr:
            return self.config.learning_rate
        frac = 1.0 - (self._update_idx / max(1, self.total_updates))
        frac = max(0.0, frac)
        lr = frac * self.config.learning_rate
        self._set_lr(lr)
        return lr
    
    # main update
    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        """
        Run num_epochs of minibatch PPO updates over 'batch'
        Returns a dict of training metrics for logging
        """
        self._update_idx += 1
        current_lr = self._anneal_lr()

        batch = batch.to(self.device)
        n = len(batch)
        if n == 0:
            return {"lr": current_lr, "n_samples": 0}
        
        indices = np.arange(n)
        minibatch_size = min(self.config.minibatch_size, n)

        # Tracking
        pg_losses, v_losses, ent_losses, approx_kls, clip_fracs = [], [], [], [], []
        early_stopped_at = None

        self.model.train()

        for epoch in range(self.config.num_epochs):
            np.random.shuffle(indices)

            for start in range(0, n, minibatch_size):
                end = min(start + minibatch_size, n)
                mb_idx = indices[start:end]
                mb = torch.from_numpy(mb_idx).long().to(self.device)

                _, new_log_prob, entropy, new_value = self.model.get_action_and_value(
                    batch.species_indices[mb],
                    batch.move_indices[mb],
                    batch.numeric_features[mb],
                    batch.context_features[mb],
                    batch.action_masks[mb],
                    action=batch.actions[mb],
                )

                old_log_prob = batch.old_log_probs[mb]
                log_ratio = new_log_prob - old_log_prob
                ratio = log_ratio.exp()

                # Approximate KL
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()
                
                # Policy loss (clipped surrogate)
                mb_adv = batch.advantages[mb]
                unclipped = -mb_adv * ratio
                clipped = -mb_adv * torch.clamp(
                    ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef
                )
                pg_loss = torch.max(unclipped, clipped).mean()

                # Value loss (clipped)
                mb_returns = batch.returns[mb]
                mb_old_values = batch.values[mb]
                v_unclipped = (new_value - mb_returns).pow(2)
                v_clipped_pred = mb_old_values + torch.clamp(
                    new_value - mb_old_values,
                    -self.config.vf_clip_coef, self.config.vf_clip_coef
                )
                v_clipped = (v_clipped_pred - mb_returns).pow(2)
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss + self.config.vf_coef * v_loss - self.config.ent_coef * ent_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)

            # Early stop on KL blowup
            if self.config.target_kl is not None and np.mean(approx_kls[-(n // minibatch_size + 1):]) > self.config.target_kl:
                early_stopped_at = epoch + 1
                break

        # Explained variance: how much of the return variance the critic captures
        with torch.no_grad():
            y_pred = batch.values.cpu().numpy()
            y_true = batch.returns.cpu().numpy()
            var_y = float(np.var(y_true))
            explained_var = float("nan") if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y

        return {
            "lr": current_lr,
            "n_samples": n,
            "pg_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(ent_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_frac": float(np.mean(clip_fracs)),
            "explained_variance": explained_var,
            "early_stopped_at_epoch": early_stopped_at,
        }