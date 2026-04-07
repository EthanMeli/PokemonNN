"""
battle_model.py
===============
Full Transformer actor-critic model for Gen 1 Pokémon RL Agent.

Architecture context:
  - 13 tokens (12 Pokémon + 1 battle context) feed a 4-layer, 4-head,
    dim-256 Transformer encoder. Self-attention enables reasoning about
    type matchups and team-level synergies across both sides.
  - Actor head: distribution over up to 9 actions (4 moves + 5 switches),
    invalid action masked to -inf.
  - Critic head: scalar state value.
  - Both share the Transformer backbone

This module implements:
  1. BattleTransformer - 4-layer Transformer encoder with positional encoding
  2. ActorHead         - action logits with invalid action masking
  3. CriticHead        - scalar state value estimate
  4. PokemonAgent      - full model: TeamEncoder + Transformer + heads

Design Decisions:
  - Pre-LayerNorm Transformer (more stable for RL training)
  - Sinusoidal positional encoding for 13 token positions
  - Actor pools the active Pokémon token + context token for action logits
  - Critic uses mean pooling over all 13 tokens for global state value
  - Action masking sets invalid logits to -inf before softmax

Usage:
  from battle_model import PokemonAgent
  from pokemon_encoder import Gen1Config

  config = Gen1Config()
  agent = PokemonAgent(config)

  # Forward pass returns policy logits and state value
  logits, value = agent(
    species_indices, move_indices, numeric_features,
    context_features, action_mask
  )
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pokemon_encoder import Gen1Config, TeamEncoder


# ==========================================================================
# Transformer Configuration (extends Gen1Config)
# ==========================================================================

@dataclass
class TransformerConfig:
  """
  Configuration for the Transformer encoder and output heads.
  Designed to work alongside Gen1Config, which handles the encoder.
  """
  # --- Transformer ---
  num_layer: int = 4          # 4 Transformer layers
  num_heads: int = 4          # 4 attention heads (256/4 = 64 per head)
  d_model: int = 256          # must match Gen1Config.d_model
  d_feedforward: int = 1024   # standard 4 x expansion (256 * 4)
  dropout: float = 0.1
  num_tokens: int = 13        # 12 Pokémon + 1 context

  # --- Action Space ---
  num_moves: int = 4          # up to 4 moves per Pokémon
  num_switches: int = 5       # up to 5 benched Pokémon to switch to
  num_actions: int = 9        # 4 moves + 5 switches

  # --- Output Heads ---
  head_hidden_dim: int = 256  # hidden dim in actor/critic head MLPs

  # --- Token position indices (convention) ---
  # tokens[:, 0:6]  = own team (slot 0 is active)
  # tokens[:, 6:12] = opponent team
  # tokens[:, 12]   = battle context
  own_active_idx: int = 0     # own active Pokémon is always at position 0
  own_team_start: int = 0
  own_team_end: int = 6
  opp_team_start: int = 6
  opp_team_end: int = 12
  context_idx: int = 12


# ==========================================================================
# Positional Encoding - Learned Embeddings
# ==========================================================================

class PositionalEmbedding(nn.Module):
  """
  Learned positional embeddings for the 13 token positions.

  A learned embedding lets each structural slot acquire its own identity
  vector during training, so the model can immediately distinguish "this
  token is our active" from "this token is opponent bench slot 3" without
  having to disentangle a sinusoidal frequency code.
  """

  def __init__(self, num_tokens: int, d_model: int, dropout: float=0.1):
    super().__init__()
    self.num_tokens = num_tokens
    self.d_model = d_model

    # nn.Embedding gives us a learned vector per integer position.
    # We use a small init (std=0.02) following GPT/BERT conventions
    # large positional embeddings can dominate the token signal early in
    # training and slow learning.
    self.pos_embedding = nn.Embedding(num_tokens, d_model)
    nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    self.dropout = nn.Dropout(dropout)

    # Pre-compute the position index buffer once. Registered as a buffer
    # so it moves with .to(device)
    position_ids = torch.arange(num_tokens, dtype=torch.long).unsqueeze(0)  # (1, num_tokens)
    self.register_buffer("position_ids", position_ids) # (1, num_tokens)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Add positional encoding to input tokens.

    Args:
      x: (batch, seq_len, d_model)

    Returns:
      (batch, seq_len, d_model) with positional encoding added.
    """
    seq_len = x.size(1)
    pos_emb = self.pos_embedding(self.position_ids[:, :seq_len]) # (1, seq_len, d_model)
    x = x + pos_emb
    return self.dropout(x)
  

# ==========================================================================
# Transformer Encoder
# ==========================================================================

class BattleTransformer(nn.Module):
  """
  Transformer encoder for processing the 13-token battle state.

  4-layer, 4-head, dim-256 Transformer encoder. Self-attention enables
  reasoning about type matchups and team-level synergies across both sides.

  Uses pre-LayerNorm (norm_first=True) for training stability in RL.
  This is the variant where LayerNorm is applied before (rather than after)
  the self-attention and feedforward sublayers, which is now the standard
  for stable training in deep Transformers.
  """

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.config = config

    self.pos_encoding = PositionalEmbedding(
      num_tokens=config.num_tokens,
      d_model=config.d_model,
      dropout=config.dropout
    )

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=config.d_model,
      nhead=config.num_heads,
      dim_feedforward=config.d_feedforward,
      dropout=config.dropout,
      activation="gelu",               # GELU is standard in modern Transformers
      batch_first=True,                # (batch, seq, dim) convention
      norm_first=True,                 # pre-LayerNorm for stability
    )

    self.transformer = nn.TransformerEncoder(
      encoder_layer=encoder_layer,
      num_layers=config.num_layer,
      enable_nested_tensor=False,      # avoid nested tensor issues with masks
    )

    # Final LayerNorm after the Transformer stack (standard with pre-norm)
    self.final_norm = nn.LayerNorm(config.d_model)

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    """
    Process 13 tokens through the Transformer encoder.

    Args:
      tokens: (batch, 13, d_model) from TeamEncoder

    Returns:
      (batch, 13, d_model) contextualized token representation
    """
    x = self.pos_encoding(tokens)
    x = self.transformer(x)
    x = self.final_norm(x)
    return x
  

# ==========================================================================
# Actor Head (Policy)
# ==========================================================================

class ActorHead(nn.Module):
  """
  Actor head: produces action logits with invalid action masking.

  From paper:
    Actor head: distribution over up to 0 actions (4 moves + 5 switches),
    invalid actions masked to -inf.

  Architecture:
    The actor concatenates the active Pokémon token with the context
    token: (batch, 2*d_model), then passes through a 2-layer MLP to
    produce 9 logits. Invalid actions are masked to -inf before returning.
  
  Action space:
    [move_1, move_2, move_3, move_4, switch_1, switch_2, switch_3, switch_4, switch_5]
       0       1       2       3        4         5         6         7         8
  """

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.config = config

    # Input: active Pokémon token (d_model) + context token (d_model)
    input_dim = config.d_model * 2

    self.mlp = nn.Sequential(
      nn.Linear(input_dim, config.head_hidden_dim),
      nn.ReLU(),
      nn.Dropout(config.dropout),
      nn.Linear(config.head_hidden_dim, config.num_actions),
    )

    self._init_weights()

  def _init_weights(self):
    """Initialize with small weights for stable early training"""
    for module in self.mlp:
      if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=0.1)
        nn.init.zeros_(module.bias)

  def forward(self, transformer_output: torch.Tensor,
              action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute action logits with optional masking.

    Args:
      transformer_output: (batch, 13, d_model) from BattleTransformer.
      action_mask: (batch, 9) boolean tensor.
                   True = action is VALID, False = action is INVALID.
                   If None, all actions are considered valid.

    Returns:
      logits: (batch, 9) raw action logits (masked invalid -> -inf).
    """
    active_token = transformer_output[:, self.config.own_active_idx, :]  # (B, d_model)
    context_token = transformer_output[:, self.config.context_idx, :]    # (B, d_model)
    actor_input = torch.cat([active_token, context_token], dim=-1)       # (B, 2*d_model)
    logits = self.mlp(actor_input) # (B, 9)

    if action_mask is not None:
      action_mask = action_mask.to(logits.device)
      logits = logits.masked_fill(~action_mask, float("-inf"))

    return logits
  

# ==========================================================================
# Critic Head (Value Function)
# ==========================================================================

class CriticHead(nn.Module):
  """
  Critic head: produces a scalar state value estimate.

  From paper:
    Critic head: scalar state value. Both share the Transformer backbone.

  Architecture:
    Mean-pool all 13 Transformer output tokens to get a global state
    representation, then pass through a 2-layer MLP to produce a scalar.

    Mean pooling (rather than using just the context token) gives the
    critic access to per-Pokémon details that the context token alone
    may not fully capture through attention.
  """

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.config = config

    self.mlp = nn.Sequential(
      nn.Linear(config.d_model, config.head_hidden_dim),
      nn.ReLU(),
      nn.Dropout(config.dropout),
      nn.Linear(config.head_hidden_dim, 1),
    )

    self._init_weights()

  def _init_weights(self):
    """Initialzie with small weights for stable value estimates"""
    for module in self.mlp:
      if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=1.0)
        nn.init.zeros_(module.bias)

  def forward(self, transformer_output: torch.Tensor) -> torch.Tensor:
    """
    Compute scalar state value.

    Args:
      transformer_output: (batch, 13, d_model) from BattleTransformer.

    Returns:
      value: (batch,) scalar state value estimates.
    """
    pooled = transformer_output.mean(dim=1)  # (B, d_model)
    value = self.mlp(pooled).squeeze(-1)     # (B,)
    return value
  

# ==========================================================================
# Full Model: PokemonAgent
# ==========================================================================

class PokemonAgent(nn.Module):
  """
  Complete Transformer actor-critic agent for Gen 1 Pokémon battles.

  Combines:
    TeamEncoder       - observation -> 13 tokens (from pokemon_encoder.py)
    BattleTransformer - 13 tokens -> 13 contextualized tokens
    ActorHead         - contextualized tokens -> actions logits (with masking)
    CriticHead        - contextualized tokens -> scalar state value

  This is the full model that gets called during both rollout collection
  and PPO gradient updates.
  """

  def __init__(self, gen1_config: Gen1Config,
               transformer_config: Optional[TransformerConfig] = None,
               species_matrix: Optional[torch.Tensor] = None,
               move_matrix: Optional[torch.Tensor] = None,
               unknown_embedding: Optional[torch.Tensor] = None):
    super().__init__()

    if transformer_config is None:
      transformer_config = TransformerConfig()

    assert gen1_config.d_model == transformer_config.d_model, \
      f"d_model mismatch: Gen1Config={gen1_config.d_model}, " \
      f"TransformerConfig={transformer_config.d_model}"
      
    self.gen1_config = gen1_config
    self.transformer_config = transformer_config

    self.encoder = TeamEncoder(
      gen1_config, species_matrix, move_matrix, unknown_embedding
    )
    self.transformer = BattleTransformer(transformer_config)
    self.actor = ActorHead(transformer_config)
    self.critic = CriticHead(transformer_config)

  def forward(self, species_indices: torch.Tensor,
              move_indices: torch.Tensor,
              numeric_features: torch.Tensor,
              context_features: torch.Tensor,
              action_mask: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full forward pass: observation -> (action logits, state value).

    Args:
      species_indices:  (batch, 12) int
      move_indicies:    (batch, 12, 4) int
      numeric_features: (batch, 12, numeric_dim) float
      context_features: (batch, context_dim) float
      action_mask:      (batch, 9) bool - True = valid action

    Returns:
      logits: (batch, 9) action logits (invalid masked to -inf)
      value:  (batch,) scalar state value estimates
    """
    tokens = self.encoder(
      species_indices, move_indices, numeric_features, context_features
    ) # (B, 13, 256)

    contextualized = self.transformer(tokens) # (B, 13, 256)

    logits = self.actor(contextualized, action_mask) # (B, 9)
    value = self.critic(contextualized)              # (B,)

    return logits, value
  
  def get_action_and_value(self, species_indices: torch.Tensor,
                           move_indices: torch.Tensor,
                           numeric_features: torch.Tensor,
                           context_features: torch.Tensor,
                           action_mask: Optional[torch.Tensor] = None,
                           action: Optional[torch.Tensor] = None,
                           ) -> Tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor]:
    """
    PPO-compatible forward pass: return action, log_prob, entropy, value

    Used during both rollout collection (action=None -> sample) and
    PPO gradient updates (action=provided -> evaluate log_prob)

    Args:
      species_indices, move_indices, numeric_features, context_features:
        Same as forward()
      action_mask: (batch, 9) bool
      action: (batch,) int, optional. If provided, compute log_prob
              of this action instead of sampling a new one.

    Returns:
      action:   (batch,) int - selected or provided action index
      log_prob: (batch,) float - log probability of the action
      entropy:  (batch,) float - entropy of the action distribution
      value:    (batch,) float - state value estimate
    """
    logits, value = self.forward(
      species_indices, move_indices, numeric_features,
      context_features, action_mask
    )

    dist = torch.distributions.Categorical(logits=logits)

    if action is None:
      action = dist.sample()

    log_prob = dist.log_prob(action)
    entropy = dist.entropy()

    return action, log_prob, entropy, value
  
  def get_value(self, species_indices: torch.Tensor,
                move_indices: torch.Tensor,
                numeric_features: torch.Tensor,
                context_features: torch.Tensor) -> torch.Tensor:
    """
    Get only the state value (used for GAE bootstrapping)

    More efficient than full forward when we only need the value.
    """
    tokens = self.encoder(
      species_indices, move_indices, numeric_features, context_features
    )
    contextualized = self.transformer(tokens)
    return self.critic(contextualized)
  
  def count_parameters(self,) -> Dict[str, int]:
    """Count parameters by component"""
    counts = {}

    encoder_counts = self.encoder.count_parameters()
    counts["encoder_trainable"] = encoder_counts["total_trainable"]
    counts["encoder_frozen"] = encoder_counts["total_frozen"]

    counts["transformer"] = sum(
      p.numel() for p in self.transformer.parameters() if p.requires_grad
    )
    counts["actor_head"] = sum(
      p.numel() for p in self.actor.parameters() if p.requires_grad
    )
    counts["critic_head"] = sum(
      p.numel() for p in self.critic.parameters() if p.requires_grad
    )

    counts["total_trainable"] = sum(
      p.numel() for p in self.parameters() if p.requires_grad
    )
    counts["total_frozen"] = encoder_counts["total_frozen"]
    counts["total_all"] = counts["total_trainable"] + counts["total_frozen"]

    return counts
  

# ==========================================================================
# Unit Tests
# ==========================================================================

def _run_unit_tests():
  """Comprehensive unit tests for the full model."""
  print("=" * 60)
  print("Battle Model - Unit Tests (Gen 1)")
  print("=" * 60)

  gen1_config = Gen1Config()
  tf_config = TransformerConfig()

  # --- Test 1: BattleTransformer standalone ---
  print("\n--- Test 1: BattleTransformer standalone ---")
  transformer = BattleTransformer(tf_config)
  dummy_tokens = torch.randn(4, 13, 256)
  out = transformer(dummy_tokens)
  assert out.shape == (4, 13, 256), f"Expected (4, 13, 256), got {out.shape}"
  print(f"  Input: {dummy_tokens.shape}")
  print(f"  Output: {out.shape}")
  print(f"  ✓ BattleTransformer forward pass correct")

  # --- Test 2: ActorHead with masking ---
  print("\n--- Test 2: ActorHead with action masking ---")
  actor = ActorHead(tf_config)
  dummy_tf_out = torch.randn(4, 13, 256)

  logits_all = actor(dummy_tf_out)
  assert logits_all.shape == (4, 9)
  assert torch.all(torch.isfinite(logits_all))
  print(f"  No mask: logits shape {logits_all.shape}, all finite ✓")

  mask = torch.ones(4, 9, dtype=torch.bool)
  mask[:, 4:] = False # disable all switches
  logits_masked = actor(dummy_tf_out, mask)
  assert torch.all(logits_masked[:, 4:] == float("-inf"))
  assert torch.all(torch.isfinite(logits_masked[:, :4]))
  print(f"  With mask: switches masked to -inf ✓")

  probs = F.softmax(logits_masked, dim=1)
  assert torch.allclose(probs[:, 4:], torch.zeros_like(probs[:, 4:]))
  print(f"  Softmax assigns zero prob to masked actions ✓")

  # --- Test 3: CriticHead ---
  print("\n--- Test 3: CriticHead ---")
  critic = CriticHead(tf_config)
  value = critic(dummy_tf_out)
  assert value.shape == (4,)
  print(f"  Value shape: {value.shape} ✓")

  # --- Test 4: Full PokemonAgent forward pass ---
  print("\n--- Test 4: Full PokemonAgent forward pass ---")
  agent = PokemonAgent(gen1_config, tf_config)

  batch_size = 8
  species_idx = torch.randint(0, gen1_config.num_species, (batch_size, 12))
  move_idx = torch.randint(0, gen1_config.num_moves, (batch_size, 12, 4))
  numeric = torch.randn(batch_size, 12, gen1_config.numeric_features_per_pokemon)
  context = torch.randn(batch_size, gen1_config.battle_context_dim)
  action_mask = torch.ones(batch_size, 9, dtype=torch.bool)
  action_mask[:, 3] = False  # move 4 unavailable
  action_mask[:, 7:] = False # switches 4,5 unavailable

  logits, value = agent(species_idx, move_idx, numeric, context, action_mask)

  assert logits.shape == (batch_size, 9)
  assert value.shape == (batch_size,)
  assert torch.all(logits[:, 3] == float("-inf"))
  assert torch.all(logits[:, 7:] == float("-inf"))
  assert torch.all(torch.isfinite(logits[:, :3]))
  assert torch.all(torch.isfinite(logits[:, 4:7]))
  print(f"  Logits shape: {logits.shape}, value shape: {value.shape}")
  print(f"  Masked actions correctly set to -inf ✓")

  # --- Test 5: PPO-compatible get_action_and_value ---
  print("\n--- Test 5: PPO get_action_and_value ---")
  action, log_prob, entropy, value = agent.get_action_and_value(
    species_idx, move_idx, numeric, context, action_mask
  )
  assert action.shape == (batch_size,)
  assert log_prob.shape == (batch_size,)
  assert entropy.shape == (batch_size,)
  assert value.shape == (batch_size,)

  for i in range(batch_size):
    assert action_mask[i, action[i].item()], \
      f"Sampled action {action[i].item()} invalid for batch {i}"
  print(f"  Actions: {action.tolist()}")
  print(f"  All sampled actions are in valid set ✓")

  # Evaluation mode: given action (must disable dropout for determinism)
  agent.eval()
  with torch.no_grad():
    _, log_prob_eval, _, _ = agent.get_action_and_value(
      species_idx, move_idx, numeric, context, action_mask
    )
    action_for_check = log_prob_eval.clone() # checking shapes

  # Re-run with same action to verify consistency in eval mode
  with torch.no_grad():
    action_sample, lp1, _, _ = agent.get_action_and_value(
      species_idx, move_idx, numeric, context, action_mask
    )
    _, lp2, _, _ = agent.get_action_and_value(
      species_idx, move_idx, numeric, context, action_mask, action=action_sample
    )
  assert torch.allclose(lp1, lp2)
  agent.train() # restore train mode for remaining tests
  print(f"  Log prob matches for given action (eval mode) ✓")

  # --- Test 6: get_value ---
  print("\n--- Test 6: get_value (GAE bootstrapping) ---")
  value_only = agent.get_value(species_idx, move_idx, numeric, context)
  assert value_only.shape == (batch_size,)
  print(f"  Value-only shape: {value_only.shape} ✓")

  # --- Test 7: Gradient flow ---
  print("\n--- Test 7: Gradient flow through entire model ---")
  agent.zero_grad()
  logits, value = agent(species_idx, move_idx, numeric, context, action_mask)

  dist = torch.distributions.Categorical(logits=logits)
  sampled = dist.sample()
  log_probs = dist.log_prob(sampled)
  fake_adv = torch.randn(batch_size)
  fake_ret = torch.randn(batch_size)

  policy_loss = -(log_probs * fake_adv).mean()
  value_loss = F.mse_loss(value, fake_ret)
  entropy_loss = -dist.entropy().mean()
  total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
  total_loss.backward()

  grad_checks = {
    "species_proj": agent.encoder.entity_embeddings.species_proj.weight.grad is not None,
    "move_proj": agent.encoder.entity_embeddings.move_proj.weight.grad is not None,
    "pokemon_mlp": agent.encoder.pokemon_encoder.mlp[0].weight.grad is not None,
    "transformer_L0": list(agent.transformer.transformer.layers[0].parameters())[0].grad is not None,
    "actor_head": agent.actor.mlp[0].weight.grad is not None,
    "critic_head": agent.critic.mlp[0].weight.grad is not None,
  }
  for name, ok in grad_checks.items():
    print(f"  {name:25s}: {'✓' if ok else 'X MISSING'}")
  assert all(grad_checks.values()), "All components should receive gradients"
  assert not agent.encoder.entity_embeddings.species_embeddings.requires_grad
  print(f"  Frozen embeddings remain frozen ✓")
  print(f"  Total loss: {total_loss.item():.4f}")

  # --- Test 8: Parameter counts ---
  print("\n--- Test 8: Parameter counts ---")
  counts = agent.count_parameters()
  for name, count in counts.items():
    print(f"  {name:25s}: {count:>12,}")
  total = counts["total_trainable"]
  assert 1_000_000 < total < 15_000_000, f"Total {total:,} outside 1M-15M range"
  print(f"\n  Total trainable: {total:,} - within target ✓")

  # --- Test 9: Force-switch scenario ---
  print("\n--- Test 9: Force-switch scenario ---")
  force_mask = torch.zeros(batch_size, 9, dtype=torch.bool)
  force_mask[:, 4:7] = True # only switches 1-3
  logits_fs, _ = agent(species_idx, move_idx, numeric, context, force_mask)
  assert torch.all(logits_fs[:, :4] == float("-inf"))
  assert torch.all(logits_fs[:, 7:] == float("-inf"))
  assert torch.all(torch.isfinite(logits_fs[:, 4:7]))
  probs_fs = F.softmax(logits_fs, dim=1)
  print(f"  Force-switch: moves={probs_fs[0, :4].sum():.3f}, "
        f"switches={probs_fs[0, 4:].sum():.3f} ✓")
  
  # --- Test 10: Device handling ---
  print("\n--- Test 10: Device handling ---")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  agent_dev = PokemonAgent(gen1_config, tf_config).to(device)
  logits_dev, value_dev = agent_dev(
    species_idx, move_idx, numeric, context, action_mask
  )
  assert logits_dev.device.type == device.type
  print(f"  Device: {device}, outputs on correct device ✓")

  # --- Test 11: Integration with real embeddings ---
  print("\n--- Test 11: Real embedding integration ---")
  import os
  emb_dir = "embeddings"
  if os.path.exists(os.path.join(emb_dir, "species_matrix.pt")):
    sp = torch.load(os.path.join(emb_dir, "species_matrix.pt"),
                    weights_only=True, map_location="cpu")
    mv = torch.load(os.path.join(emb_dir, "move_matrix.pt"),
                    weights_only=True, map_location="cpu")
    unk = torch.load(os.path.join(emb_dir, "unknown_embedding.pt"),
                     weights_only=True, map_location="cpu")
    real_agent = PokemonAgent(gen1_config, tf_config, sp, mv, unk).to(device)
    logits_r, value_r = real_agent(
      species_idx, move_idx, numeric, context, action_mask
    )
    assert logits_r.shape == (batch_size, 9)
    print(f"  ✓ Full model with real LLM embeddings works end-to-end")
  else:
    print(f"  Embeddings not found. Run generate_embeddings.py first")

  print(f"\n{'=' * 60}")
  print(f"All unit tests passed!")
  print(f"{'=' * 60}")

if __name__ == "__main__":
  _run_unit_tests()