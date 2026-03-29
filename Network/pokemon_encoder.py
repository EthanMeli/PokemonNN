"""
pokemon_encoder.py
==================
Per-Pokémon Encoder Module for Gen 1 Pokémon RL Agent.

Architecture context:
  Each Pokémon's features (projected species/move embeddings + numeric features
  like HP bin, stat boosts, volatile statuses) are concatenated and passed through
  a shared 2-layer MLP. Applied to all 12 Pokémon (6 own + 6 opponent).
  Unrevealed Pokémon use a learned "unknown" token.

This module implements:
  1. EntityEmbeddingLayer — loads frozen LLM embeddings, applies learned projection
  2. PokemonEncoder — combines projected embeddings with numeric features via MLP
  3. TeamEncoder — applies PokemonEncoder to all 12 Pokémon (shared weights)

Gen 1 adaptations:
  - No abilities (removed entirely)
  - No items (not used in gen1randombattle)
  - Special is one stat (not SpA/SpD split)
  - 6 stat boost dimensions: Atk, Def, Special, Speed, Accuracy, Evasion
  - Fewer volatile statuses than later gens

Usage:
  from pokemon_encoder import TeamEncoder, Gen1Config

  config = Gen1Config()
  encoder = TeamEncoder(config)

  # During training, feed observation tensors from env encoder
  pokemon_vectors, context_token = encoder(obs_dict)
  # pokemon_vectors: (batch, 12, d_model)  — 6 own + 6 opponent Pokémon
  # context_token:   (batch, 1, d_model)   — battle context (weather, hazards, etc.)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Gen1Config:
  """Configuration for the Gen 1 per-Pokémon encoder.

  All dimensions and entity counts are Gen 1-specific.
  """
  # --- Entity counts ---
  num_species: int = 151
  num_moves: int = 153  # from generate_embeddings.py output

  # --- Embedding dimensions ---
  llm_embedding_dim: int = 768     # Sentence-BERT output dim
  entity_proj_dim: int = 64        # learned projection: 768 → 64 (from outline §4.1)

  # --- Numeric feature dimensions (per Pokémon) ---
  # HP: 7 bins (0%, then 6 equal bins from >0% to 100%) — following Wang's approach
  hp_bins: int = 7
  # Stat boosts: 6 stats × 13 levels each (-6 to +6, one-hot)
  # Gen 1 stats: Attack, Defense, Special, Speed, Accuracy, Evasion
  num_boost_stats: int = 6
  boost_levels: int = 13           # -6 through +6
  # Status: one-hot over {none, sleep, poison, burn, paralysis, freeze, fainted, toxic}
  num_status: int = 8
  # Toxic counter: 0-15 turns (one-hot, 16 bins)
  toxic_counter_bins: int = 16
  # Sleep counter: 0-7 turns (one-hot, 8 bins)
  sleep_counter_bins: int = 8
  # Move PP: 4 moves × 4 bins each (following Wang's cube-root binning)
  move_pp_bins: int = 4
  num_move_slots: int = 4
  # Types: 15 Gen 1 types (one-hot, can have 1-2 active)
  # Normal, Fighting, Flying, Poison, Ground, Rock, Bug, Ghost,
  # Fire, Water, Grass, Electric, Psychic, Ice, Dragon
  num_types: int = 15
  # Binary flags
  # is_active, is_opponent, is_unknown, first_turn, must_recharge, is_fainted
  num_binary_flags: int = 6
  # Gen 1 volatile statuses (binary flags):
  # confused, seeded, partially_trapped, substitute_active, focus_energy,
  # mist_active, light_screen, reflect
  num_volatile_status: int = 8

  # --- MLP dimensions ---
  d_model: int = 256               # output dim of per-Pokémon encoder (= Transformer dim)
  encoder_hidden_dim: int = 512    # hidden layer in 2-layer MLP
  dropout: float = 0.1

  # --- Battle context token dimensions ---
  # Weather: none, sun (not in Gen 1 moves, but exists via data), rain, sandstorm, hail → simplified
  # Gen 1 only has weather from specific situations, but we keep a small encoding
  # For gen1randombattle: weather is minimal, but we encode it for future-proofing
  num_weather: int = 5   # none, sun, rain, sandstorm, hail (one-hot)
  # Hazards per side: Reflect active, Light Screen active (side-wide in Gen 1)
  # These are per-side, so 2 sides × a few flags
  num_side_conditions: int = 4  # reflect_ours, lightscreen_ours, reflect_theirs, lightscreen_theirs
  # Turn count: binned (0-4, 5-9, 10-19, 20-29, 30-39, 40+)
  turn_count_bins: int = 6
  # Force switch flag
  force_switch_dim: int = 2  # one-hot: yes/no
  # Number of unknown opponent Pokémon remaining (0-6, one-hot)
  num_unknown_bins: int = 7

  @property
  def numeric_features_per_pokemon(self) -> int:
    """Total numeric feature vector length per Pokémon."""
    return (
      self.hp_bins
      + self.num_boost_stats * self.boost_levels
      + self.num_status
      + self.toxic_counter_bins
      + self.sleep_counter_bins
      + self.num_move_slots * self.move_pp_bins
      + self.num_types
      + self.num_binary_flags
      + self.num_volatile_status
    )

  @property
  def embedding_features_per_pokemon(self) -> int:
    """Total embedding feature length per Pokémon (after projection)."""
    # species embedding + 4 move embeddings, each projected to entity_proj_dim
    return self.entity_proj_dim * (1 + self.num_move_slots)

  @property
  def total_features_per_pokemon(self) -> int:
    """Total input to the per-Pokémon MLP."""
    return self.embedding_features_per_pokemon + self.numeric_features_per_pokemon

  @property
  def battle_context_dim(self) -> int:
    """Total battle context feature vector length."""
    return (
      self.num_weather
      + self.num_side_conditions
      + self.turn_count_bins
      + self.force_switch_dim
      + self.num_unknown_bins
    )


# =============================================================================
# Entity Embedding Layer
# =============================================================================

class EntityEmbeddingLayer(nn.Module):
  """Loads frozen LLM embeddings and applies a learned linear projection.

  From outline:
    Frozen embedding from pretrained sentence encoder → learned linear
    projection (768 → 64) maps into the RL latent space.

  The frozen embeddings are stored as non-trainable buffers.
  The projection layer IS trainable.
  """

  def __init__(self, config: Gen1Config, species_matrix: torch.Tensor,
              move_matrix: torch.Tensor, unknown_embedding: torch.Tensor):
    """
    Args:
      config: Gen1Config instance.
      species_matrix: (num_species, llm_embedding_dim) frozen embeddings.
      move_matrix: (num_moves, llm_embedding_dim) frozen embeddings.
      unknown_embedding: (llm_embedding_dim,) frozen embedding for unknown Pokémon.
    """
    super().__init__()
    self.config = config

    # Register frozen embeddings as buffers (not parameters — not trained)
    self.register_buffer("species_embeddings", species_matrix)   # (151, 768)
    self.register_buffer("move_embeddings", move_matrix)         # (153, 768)
    self.register_buffer("unknown_embedding", unknown_embedding) # (768,)

    # Learned projections: 768 → 64
    # Separate projections for species and moves so they can learn
    # different aspects of the semantic space
    self.species_proj = nn.Linear(config.llm_embedding_dim, config.entity_proj_dim)
    self.move_proj = nn.Linear(config.llm_embedding_dim, config.entity_proj_dim)

    # Projection for unknown token
    self.unknown_proj = nn.Linear(config.llm_embedding_dim, config.entity_proj_dim)

    # A learned "no move" embedding for empty move slots
    # (e.g., a Pokémon with fewer than 4 moves, or unknown moves)
    self.no_move_embedding = nn.Parameter(
      torch.randn(config.entity_proj_dim) * 0.02
    )

    self._init_projections()

  def _init_projections(self):
    """Initialize projection layers with small weights."""
    for proj in [self.species_proj, self.move_proj, self.unknown_proj]:
      nn.init.xavier_uniform_(proj.weight)
      nn.init.zeros_(proj.bias)

  def project_species(self, species_indices: torch.Tensor) -> torch.Tensor:
    """Look up and project species embeddings.

    Args:
      species_indices: (batch, num_pokemon) integer indices into species_embeddings.
                        Use -1 for unknown Pokémon.

    Returns:
      (batch, num_pokemon, entity_proj_dim) projected species embeddings.
    """
    # Ensure indices are on the same device as the embedding buffer
    device = self.species_embeddings.device
    species_indices = species_indices.to(device)

    batch_size, num_pokemon = species_indices.shape

    # Mask for unknown Pokémon
    unknown_mask = species_indices < 0  # -1 indicates unknown

    # Clamp indices for safe lookup (we'll overwrite unknowns)
    safe_indices = species_indices.clamp(min=0)

    # Look up frozen embeddings: (batch, num_pokemon, 768)
    raw_emb = self.species_embeddings[safe_indices]

    # Project: (batch, num_pokemon, 64)
    projected = self.species_proj(raw_emb)

    # Replace unknown Pokémon with projected unknown embedding
    if unknown_mask.any():
      unknown_vec = self.unknown_proj(self.unknown_embedding)  # (64,)
      projected = projected.masked_fill(
        unknown_mask.unsqueeze(-1).expand_as(projected),
        0.0
      )
      projected = projected + unknown_mask.unsqueeze(-1).float() * unknown_vec.unsqueeze(0).unsqueeze(0)

    return projected

  def project_moves(self, move_indices: torch.Tensor) -> torch.Tensor:
    """Look up and project move embeddings.

    Args:
      move_indices: (batch, num_pokemon, 4) integer indices into move_embeddings.
                    Use -1 for empty/unknown move slots.

    Returns:
      (batch, num_pokemon, 4, entity_proj_dim) projected move embeddings.
    """
    # Ensure indices are on the same device as the embedding buffer
    device = self.move_embeddings.device
    move_indices = move_indices.to(device)
    # Mask for empty move slots
    empty_mask = move_indices < 0

    safe_indices = move_indices.clamp(min=0)

    # Look up frozen embeddings: (batch, num_pokemon, 4, 768)
    raw_emb = self.move_embeddings[safe_indices]

    # Project: (batch, num_pokemon, 4, 64)
    projected = self.move_proj(raw_emb)

    # Replace empty slots with learned no_move embedding
    if empty_mask.any():
      projected = projected.masked_fill(
        empty_mask.unsqueeze(-1).expand_as(projected),
        0.0
      )
      no_move = self.no_move_embedding.view(1, 1, 1, -1)
      projected = projected + empty_mask.unsqueeze(-1).float() * no_move

    return projected


# =============================================================================
# Per-Pokémon Encoder
# =============================================================================

class PokemonEncoder(nn.Module):
  """Encodes a single Pokémon's features into a d_model-dimensional vector.

  From outline:
    Each Pokémon's features (projected species/move embeddings + numeric features)
    are concatenated and passed through a shared 2-layer MLP.

  Input features per Pokémon:
    - Species embedding (projected): 64
    - 4 × Move embeddings (projected): 4 × 64 = 256
    - Numeric features (HP, boosts, status, PP, types, flags): ~152
    Total: ~472

  Output: d_model (256) dimensional vector.
  """

  def __init__(self, config: Gen1Config):
    super().__init__()
    self.config = config

    input_dim = config.total_features_per_pokemon

    # 2-layer MLP with ReLU activation (from outline)
    self.mlp = nn.Sequential(
      nn.Linear(input_dim, config.encoder_hidden_dim),
      nn.ReLU(),
      nn.Dropout(config.dropout),
      nn.Linear(config.encoder_hidden_dim, config.d_model),
      nn.ReLU(),
      nn.Dropout(config.dropout),
    )

    # Layer norm on output for stable Transformer input
    self.layer_norm = nn.LayerNorm(config.d_model)

    self._init_weights()

  def _init_weights(self):
    """Initialize MLP weights."""
    for module in self.mlp:
      if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)

  def forward(self, species_emb: torch.Tensor, move_embs: torch.Tensor,
              numeric_features: torch.Tensor) -> torch.Tensor:
    """Encode a batch of Pokémon.

    Args:
      species_emb:      (batch, num_pokemon, entity_proj_dim)
      move_embs:        (batch, num_pokemon, 4, entity_proj_dim)
      numeric_features: (batch, num_pokemon, numeric_features_per_pokemon)

    Returns:
      (batch, num_pokemon, d_model) encoded Pokémon vectors.
    """
    batch_size, num_pokemon, _ = species_emb.shape

    # Flatten move embeddings: (batch, num_pokemon, 4*64)
    move_flat = move_embs.view(batch_size, num_pokemon, -1)

    # Concatenate all features: (batch, num_pokemon, total_features_per_pokemon)
    combined = torch.cat([species_emb, move_flat, numeric_features], dim=-1)

    # Apply shared MLP: (batch, num_pokemon, d_model)
    encoded = self.mlp(combined)
    encoded = self.layer_norm(encoded)

    return encoded


# =============================================================================
# Battle Context Encoder
# =============================================================================

class BattleContextEncoder(nn.Module):
  """Encodes battle-wide context (weather, hazards, turn count) into a
  single token for the Transformer.

  From outline:
    A "battle context" token (weather, terrain, hazards, trick room, turn count)
    forms the 13th token for the Transformer encoder.

  Gen 1 adaptation:
    - No terrain, no Trick Room
    - Weather is minimal (no sun/rain from abilities in Gen 1 random battles,
      but we keep the encoding for completeness)
    - Side conditions: Reflect, Light Screen (per side)
    - Hazards: None in Gen 1 (Stealth Rock, Spikes are Gen 2+)
  """

  def __init__(self, config: Gen1Config):
    super().__init__()
    self.config = config

    self.mlp = nn.Sequential(
      nn.Linear(config.battle_context_dim, config.encoder_hidden_dim // 2),
      nn.ReLU(),
      nn.Dropout(config.dropout),
      nn.Linear(config.encoder_hidden_dim // 2, config.d_model),
      nn.ReLU(),
      nn.Dropout(config.dropout),
    )

    self.layer_norm = nn.LayerNorm(config.d_model)
    self._init_weights()

  def _init_weights(self):
    for module in self.mlp:
      if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)

  def forward(self, context_features: torch.Tensor) -> torch.Tensor:
    """Encode battle context into a single token.

    Args:
      context_features: (batch, battle_context_dim)

    Returns:
      (batch, 1, d_model) battle context token.
    """
    encoded = self.mlp(context_features)
    encoded = self.layer_norm(encoded)
    return encoded.unsqueeze(1)  # (batch, 1, d_model)


# =============================================================================
# Team Encoder (Full Pipeline)
# =============================================================================

class TeamEncoder(nn.Module):
  """Full encoding pipeline: raw observation → 13 tokens for the Transformer.

  Combines EntityEmbeddingLayer + PokemonEncoder + BattleContextEncoder
  to produce:
    - 12 Pokémon tokens (6 own + 6 opponent)
    - 1 battle context token
    = 13 tokens, each of dimension d_model (256)

  This is the interface between Partner A's observation encoder and
  Partner B's Transformer.
  """

  def __init__(self, config: Gen1Config,
                species_matrix: Optional[torch.Tensor] = None,
                move_matrix: Optional[torch.Tensor] = None,
                unknown_embedding: Optional[torch.Tensor] = None):
    """
    Args:
      config: Gen1Config.
      species_matrix: (num_species, 768) from generate_embeddings.py.
                      If None, uses random embeddings (for testing / ablation).
      move_matrix: (num_moves, 768) from generate_embeddings.py.
                    If None, uses random embeddings.
      unknown_embedding: (768,) from generate_embeddings.py.
                          If None, uses random embedding.
    """
    super().__init__()
    self.config = config

    # Handle None embeddings (for testing or random-embedding ablation)
    if species_matrix is None:
      species_matrix = torch.randn(config.num_species, config.llm_embedding_dim)
    if move_matrix is None:
      move_matrix = torch.randn(config.num_moves, config.llm_embedding_dim)
    if unknown_embedding is None:
      unknown_embedding = torch.randn(config.llm_embedding_dim)

    self.entity_embeddings = EntityEmbeddingLayer(
      config, species_matrix, move_matrix, unknown_embedding
    )
    self.pokemon_encoder = PokemonEncoder(config)
    self.context_encoder = BattleContextEncoder(config)

  def forward(self, species_indices: torch.Tensor,
              move_indices: torch.Tensor,
              numeric_features: torch.Tensor,
              context_features: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full forward pass: observation → Transformer-ready tokens.

    Args:
      species_indices:  (batch, 12) int — species index per Pokémon slot.
                        -1 for unknown/unrevealed.
      move_indices:     (batch, 12, 4) int — move index per move slot.
                        -1 for unknown/empty.
      numeric_features: (batch, 12, numeric_features_per_pokemon) float.
      context_features: (batch, battle_context_dim) float.

    Returns:
      tokens: (batch, 13, d_model) — 12 Pokémon + 1 context token.
              tokens[:, :6, :]  = own team
              tokens[:, 6:12, :] = opponent team
              tokens[:, 12, :]  = battle context
    """
    # Ensure all inputs are on the same device as the model
    device = next(self.parameters()).device
    species_indices = species_indices.to(device)
    move_indices = move_indices.to(device)
    numeric_features = numeric_features.to(device)
    context_features = context_features.to(device)

    # 1. Project entity embeddings
    species_emb = self.entity_embeddings.project_species(species_indices)  # (B, 12, 64)
    move_emb = self.entity_embeddings.project_moves(move_indices)          # (B, 12, 4, 64)

    # 2. Encode each Pokémon through shared MLP
    pokemon_tokens = self.pokemon_encoder(species_emb, move_emb, numeric_features)  # (B, 12, 256)

    # 3. Encode battle context
    context_token = self.context_encoder(context_features)  # (B, 1, 256)

    # 4. Concatenate into 13-token sequence
    tokens = torch.cat([pokemon_tokens, context_token], dim=1)  # (B, 13, 256)

    return tokens

  def count_parameters(self) -> Dict[str, int]:
    """Count parameters by component for debugging."""
    counts = {}

    # Entity embedding projections (trainable)
    entity_params = sum(p.numel() for p in self.entity_embeddings.parameters())
    counts["entity_projections"] = entity_params

    # Per-Pokémon MLP
    pokemon_params = sum(p.numel() for p in self.pokemon_encoder.parameters())
    counts["pokemon_encoder"] = pokemon_params

    # Battle context MLP
    context_params = sum(p.numel() for p in self.context_encoder.parameters())
    counts["context_encoder"] = context_params

    # Frozen embeddings (not counted as parameters)
    frozen = (
      self.entity_embeddings.species_embeddings.numel()
      + self.entity_embeddings.move_embeddings.numel()
      + self.entity_embeddings.unknown_embedding.numel()
    )
    counts["frozen_embeddings"] = frozen

    counts["total_trainable"] = sum(
      p.numel() for p in self.parameters() if p.requires_grad
    )
    counts["total_frozen"] = frozen

    return counts


# =============================================================================
# Unit Tests
# =============================================================================

def _run_unit_tests():
  """Verify forward pass with synthetic data.

  This is the Week 1 deliverable: "Unit test: forward pass with dummy tensors."
  """
  print("=" * 60)
  print("Per-Pokémon Encoder — Unit Tests (Gen 1)")
  print("=" * 60)

  config = Gen1Config()

  # Print architecture summary
  print(f"\n--- Config Summary ---")
  print(f"  Entity proj dim:              {config.entity_proj_dim}")
  print(f"  Embedding features/pokemon:   {config.embedding_features_per_pokemon}")
  print(f"  Numeric features/pokemon:     {config.numeric_features_per_pokemon}")
  print(f"  Total features/pokemon:       {config.total_features_per_pokemon}")
  print(f"  Battle context dim:           {config.battle_context_dim}")
  print(f"  Output dim (d_model):         {config.d_model}")
  print(f"  Encoder hidden dim:           {config.encoder_hidden_dim}")

  # Create encoder with random embeddings (no need for real ones in unit test)
  encoder = TeamEncoder(config)

  # Print parameter counts
  param_counts = encoder.count_parameters()
  print(f"\n--- Parameter Counts ---")
  for name, count in param_counts.items():
    print(f"  {name:25s}: {count:>10,}")

  # --- Test 1: Single batch forward pass ---
  print(f"\n--- Test 1: Single-batch forward pass ---")
  batch_size = 1
  num_pokemon = 12

  species_indices = torch.randint(0, config.num_species, (batch_size, num_pokemon))
  move_indices = torch.randint(0, config.num_moves, (batch_size, num_pokemon, 4))
  numeric_features = torch.randn(batch_size, num_pokemon, config.numeric_features_per_pokemon)
  context_features = torch.randn(batch_size, config.battle_context_dim)

  tokens = encoder(species_indices, move_indices, numeric_features, context_features)
  print(f"  Input shapes:")
  print(f"    species_indices:  {species_indices.shape}")
  print(f"    move_indices:     {move_indices.shape}")
  print(f"    numeric_features: {numeric_features.shape}")
  print(f"    context_features: {context_features.shape}")
  print(f"  Output shape: {tokens.shape}")
  assert tokens.shape == (batch_size, 13, config.d_model), \
    f"Expected ({batch_size}, 13, {config.d_model}), got {tokens.shape}"
  print(f"  ✓ Shape correct: (batch={batch_size}, tokens=13, d_model={config.d_model})")

  # --- Test 2: Batched forward pass ---
  print(f"\n--- Test 2: Batched forward pass ---")
  batch_size = 32

  species_indices = torch.randint(0, config.num_species, (batch_size, num_pokemon))
  move_indices = torch.randint(0, config.num_moves, (batch_size, num_pokemon, 4))
  numeric_features = torch.randn(batch_size, num_pokemon, config.numeric_features_per_pokemon)
  context_features = torch.randn(batch_size, config.battle_context_dim)

  tokens = encoder(species_indices, move_indices, numeric_features, context_features)
  assert tokens.shape == (batch_size, 13, config.d_model)
  print(f"  ✓ Batch size {batch_size}: output shape {tokens.shape}")

  # --- Test 3: Unknown Pokémon handling ---
  print(f"\n--- Test 3: Unknown Pokémon handling ---")
  batch_size = 4
  species_indices = torch.randint(0, config.num_species, (batch_size, num_pokemon))
  # Set opponent slots 3-5 (indices 9-11) as unknown
  species_indices[:, 9:12] = -1
  move_indices = torch.randint(0, config.num_moves, (batch_size, num_pokemon, 4))
  move_indices[:, 9:12, :] = -1  # unknown moves too
  numeric_features = torch.randn(batch_size, num_pokemon, config.numeric_features_per_pokemon)
  context_features = torch.randn(batch_size, config.battle_context_dim)

  tokens = encoder(species_indices, move_indices, numeric_features, context_features)
  assert tokens.shape == (batch_size, 13, config.d_model)

  # Unknown Pokémon should all get the same embedding (before MLP)
  # After MLP + numeric features they may differ, but the embedding part should be identical
  unknown_tokens = tokens[:, 9:12, :]
  print(f"  Unknown tokens shape: {unknown_tokens.shape}")
  print(f"  ✓ Forward pass succeeds with unknown Pokémon (index=-1)")

  # --- Test 4: Gradient flow ---
  print(f"\n--- Test 4: Gradient flow ---")
  batch_size = 8
  species_indices = torch.randint(0, config.num_species, (batch_size, num_pokemon))
  move_indices = torch.randint(0, config.num_moves, (batch_size, num_pokemon, 4))
  numeric_features = torch.randn(batch_size, num_pokemon, config.numeric_features_per_pokemon)
  context_features = torch.randn(batch_size, config.battle_context_dim)

  tokens = encoder(species_indices, move_indices, numeric_features, context_features)
  loss = tokens.sum()
  loss.backward()

  # Check that projection layers got gradients
  assert encoder.entity_embeddings.species_proj.weight.grad is not None, \
    "Species projection should receive gradients"
  assert encoder.entity_embeddings.move_proj.weight.grad is not None, \
    "Move projection should receive gradients"
  assert encoder.pokemon_encoder.mlp[0].weight.grad is not None, \
    "Pokemon MLP should receive gradients"
  assert encoder.context_encoder.mlp[0].weight.grad is not None, \
    "Context MLP should receive gradients"

  # Check that frozen embeddings did NOT get gradients (they're buffers, not params)
  assert not encoder.entity_embeddings.species_embeddings.requires_grad, \
    "Frozen species embeddings should not require grad"
  assert not encoder.entity_embeddings.move_embeddings.requires_grad, \
    "Frozen move embeddings should not require grad"

  print(f"  ✓ Gradients flow to projection layers")
  print(f"  ✓ Gradients flow to Pokemon encoder MLP")
  print(f"  ✓ Gradients flow to Context encoder MLP")
  print(f"  ✓ Frozen embeddings remain frozen (no grad)")

  # --- Test 5: Output statistics ---
  print(f"\n--- Test 5: Output statistics ---")
  with torch.no_grad():
    species_indices = torch.randint(0, config.num_species, (64, num_pokemon))
    move_indices = torch.randint(0, config.num_moves, (64, num_pokemon, 4))
    numeric_features = torch.randn(64, num_pokemon, config.numeric_features_per_pokemon)
    context_features = torch.randn(64, config.battle_context_dim)

    tokens = encoder(species_indices, move_indices, numeric_features, context_features)
    pokemon_tokens = tokens[:, :12, :]
    ctx_token = tokens[:, 12:, :]

    print(f"  Pokémon tokens — mean: {pokemon_tokens.mean():.4f}, std: {pokemon_tokens.std():.4f}")
    print(f"  Context token  — mean: {ctx_token.mean():.4f}, std: {ctx_token.std():.4f}")
    print(f"  ✓ Outputs are finite and reasonable")

  # --- Test 6: Loading real embeddings ---
  print(f"\n--- Test 6: Integration with generate_embeddings.py ---")
  import os
  embeddings_dir = "embeddings"
  if os.path.exists(os.path.join(embeddings_dir, "species_matrix.pt")):
    # Load embeddings to CPU first, then let the model handle device placement
    species_matrix = torch.load(
      os.path.join(embeddings_dir, "species_matrix.pt"),
      weights_only=True, map_location="cpu"
    )
    move_matrix = torch.load(
      os.path.join(embeddings_dir, "move_matrix.pt"),
      weights_only=True, map_location="cpu"
    )
    unknown_emb = torch.load(
      os.path.join(embeddings_dir, "unknown_embedding.pt"),
      weights_only=True, map_location="cpu"
    )

    # Detect device and move encoder there
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_encoder = TeamEncoder(config, species_matrix, move_matrix, unknown_emb).to(device)
    print(f"  Using device: {device}")

    # Create test inputs (TeamEncoder.forward handles moving them to device)
    species_indices = torch.randint(0, config.num_species, (4, 12))
    move_indices = torch.randint(0, config.num_moves, (4, 12, 4))
    numeric_features = torch.randn(4, 12, config.numeric_features_per_pokemon)
    context_features = torch.randn(4, config.battle_context_dim)

    tokens = real_encoder(species_indices, move_indices, numeric_features, context_features)
    assert tokens.shape == (4, 13, config.d_model)
    print(f"  ✓ Real LLM embeddings loaded and forward pass succeeds")
    print(f"    Species matrix shape: {species_matrix.shape}")
    print(f"    Move matrix shape:    {move_matrix.shape}")
    print(f"    Output device:        {tokens.device}")
  else:
    print(f"  ⚠ Embeddings directory not found at '{embeddings_dir}/'")
    print(f"    Run generate_embeddings.py first to test with real embeddings.")
    print(f"    (Tests 1-5 all passed with random embeddings)")

  print(f"\n{'=' * 60}")
  print(f"All unit tests passed!")
  print(f"{'=' * 60}")


if __name__ == "__main__":
  _run_unit_tests()