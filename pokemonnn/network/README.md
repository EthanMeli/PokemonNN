# Neural Network Architecture — Gen 1 Pokémon RL Agent

A Transformer actor-critic network with **LLM-initialized entity embeddings** for competitive Pokémon battling in the Generation 1 `gen1randombattle` format, trained via PPO self-play.

> **Status:** Entity embeddings and per-Pokémon encoder complete. Transformer encoder and output heads in progress.

---

## Table of Contents

- [Overview](#overview)
- [Architecture at a Glance](#architecture-at-a-glance)
- [1. Entity Embedding Layer](#1-entity-embedding-layer)
  - [1.1 Embedding Generation Pipeline](#11-embedding-generation-pipeline)
  - [1.2 Entity Counts (Gen 1)](#12-entity-counts-gen-1)
  - [1.3 Description Engineering](#13-description-engineering)
  - [1.4 Learned Projection](#14-learned-projection)
  - [1.5 Embedding Verification](#15-embedding-verification)
- [2. Per-Pokémon Encoder](#2-per-pokémon-encoder)
  - [2.1 Per-Pokémon Feature Representation](#21-per-pokémon-feature-representation)
  - [2.2 Battle Context Representation](#22-battle-context-representation)
  - [2.3 Encoding Pipeline](#23-encoding-pipeline)
  - [2.4 Parameter Counts](#24-parameter-counts)
- [3. Transformer Encoder](#3-transformer-encoder)
  - [3.1 Architecture Details](#31-architecture-details)
  - [3.2 Positional Embeddingds](#32-positional-embeddings)
  - [3.3 Why Self-Attention Matters for Pokémon](#33-why-self-attention-matters-for-pokémon)
- [4. Actor-Critic Output Heads](#4-actor-critic-output-heads)
  - [4.1 Actor Head (Policy)](#41-actor-head-policy)
  - [4.2 Action Masking](#42-action-masking)
  - [4.3 Critic Head (Value)](#43-critic-head-value)
  - [4.4 Full Model Parameter Summary](#44-full-model-parameter-summary)
- [5. Gen 1 Adaptations](#5-gen-1-adaptations)
- [File Structure](#file-structure)
- [Quick Start](#quick-start)
- [References](#references)

---

## Overview

This repository contains the **network architecture** for a reinforcement learning agent that plays Generation 1 Pokémon random battles on Pokémon Showdown. The core novelty is combining **semantic embeddings from a pretrained language model** with a **relational Transformer backbone** — an approach suggested as future work by Wang (2024) and not yet explored in any published Pokémon RL system.

The architecture processes a battle state (12 Pokémon + battle context) into 13 tokens for a Transformer encoder, which feeds shared actor-critic heads for PPO training.

### Why Gen 1?

Generation 1 offers a unique challenge: no abilities, no items in random battles, a unified Special stat, and a metagame dominated by Psychic types with [well-documented quirks](https://www.smogon.com/dex/rb/formats/ou/) (e.g., the Ghost&rarr;Psychic bug, Focus Energy *reducing* crit rate, trapping moves completely immobilizing targets). The reduced entity count (151 species, ~153 moves) allows faster iteration while preserving the core complexity of imperfect-information, simultaneous-move gameplay.

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BATTLE OBSERVATION                           │
│  species indices ─────┐                                             │
│  move indices ────────┤                                             │
│  numeric features ────┤  × 12 Pokémon (6 own + 6 opponent)          │
│  context features ────┘  + 1 battle context                         │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ENTITY EMBEDDING LAYER                           │
│                                                                     │
│  Frozen LLM Embeddings (768-dim, Sentence-BERT)                     │
│       │                                                             │
│       ▼                                                             │
│  Learned Linear Projection (768 → 64)                               │
│  • Species projection    • Move projection    • Unknown projection  │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PER-POKÉMON ENCODER                              │
│                                                                     │
│  [Species Emb (64)] + [4× Move Emb (256)] + [Numeric (162)]         │
│       │                    = 482 features                           │
│       ▼                                                             │
│  Shared 2-Layer MLP (482 → 512 → 256) + LayerNorm                   │
│                                                                     │
│  Applied identically to all 12 Pokémon (shared weights)             │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    13 TOKENS (each 256-dim)                         │
│                                                                     │
│  [Pkmn 1] [Pkmn 2] ... [Pkmn 6] [Opp 1] ... [Opp 6] [Context]       │
│   ──── own team ────    ──── opponent team ────        battle ctx   │
└────────────┬────────────────────────────────────────────────────────┘
             │ + Learned Positional Embeddings (13 x 256)
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              TRANSFORMER ENCODER                                    │
│              4-layer, 4-head, dim-256, pre-norm, GELU               │
|              Full bidirectional self-attention (~2.1M params)       |
|                                                                     |
|              Each token attends to all 13 tokens:                   |
|              own team <--> opponent team <--> battle context        |
└───────────┬──────────────────────────────────┬──────────────────────┘
            |                                  |
     Token 0 (active)                 Head pool (all 13)
            |                                  |
            ▼                                  ▼
┌────────────────────────┐      ┌────────────────────────────┐
│       ACTOR HEAD       │      │        CRITIC HEAD         │
│                        │      │                            │
│  MLP(256 → 256 → 9)    │      │  MLP(256 → 256 → 1)        │
│         │              │      │         │                  │
│    Action Masking      │      │    Scalar Value V(s)       │
│  (invalid → -inf)      │      │                            │
│         │              │      │  Used for GAE advantage    │
│         ▼              │      │  estimation in PPO         │
│  Categorical dist      │      │                            │
│  over 9 actions        │      │                            │
│  (4 moves + 5 switch)  │      │                            │
└────────────────────────┘      └────────────────────────────┘
```

---

## 1. Entity Embedding Layer

Every Gen 1 entity (species and move) is represented by a **frozen 768-dimensional embedding** from a pretrained sentence encoder, then mapped into the RL latent space by a **learned linear projection**.

This is the central novelty of our architecture. Prior Pokémon RL work (Wang 2024, Chen & Lin 2018) used either randomly initialized `nn.Embedding` layers or Node2Vec graph embeddings. Language model embeddings encode *semantic knowledge* about what each entity does — for example, the encoder should understand that Thunderbolt and Thunder are variants of the same concept, or that Alakazam and Starmie fill similar competitive roles.

### 1.1 Embedding Generation Pipeline

```
Entity Name + Stats + Description
        │
        ▼
┌──────────────────────────┐
│  Description Builder     │
│  "Charizard is a Fire/   │
│   Flying-type Pokémon    │→  Rich natural-language string
│   in Generation 1. It    │     (~100 words per entity)
│   has base stats of..."  │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Sentence-BERT Encoder   │
│  (all-mpnet-base-v2)     │→  768-dim frozen embedding
│                          │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  .pt Tensor Files        │
│  species_matrix.pt       │→  (151, 768)
│  move_matrix.pt          │→  (153, 768)
│  unknown_embedding.pt    │→  (768,)
└──────────────────────────┘
```

**Implementation:** `generate_embeddings.py`

### 1.2 Entity Counts (Gen 1)

| Entity Type | Count | Embedding Dim | Notes |
|:---|:---:|:---:|:---|
| Species | 151 | 768 | All Gen 1 Pokémon (Bulbasaur–Mew) |
| Moves | 153 | 768 | All moves learnable in `gen1randombattle` |
| Abilities | — | — | *Do not exist in Gen 1* |
| Items | — | — | *Not used in `gen1randombattle`* |
| Unknown Token | 1 | 768 | For unrevealed opponent Pokémon |

Compared to Wang's Gen 4 setup (296 species, ~200 moves, ~100 abilities, ~40 items), Gen 1 is substantially leaner. The absence of abilities and items removes two entire embedding lookup categories.

### 1.3 Description Engineering

Each entity gets a rich text description that encodes multiple dimensions of information for the sentence encoder.

**Species descriptions** incorporate:

| Component | Example (Charizard) |
|:---|:---|
| Name & typing | "Charizard is a Fire/Flying-type Pokémon in Generation 1." |
| Full base stats | "It has base stats of 78 HP, 84 Attack, 78 Defense, 109 Special, and 100 Speed (total 534)." |
| Stat profile | "Its strongest stats are Special 109 and Speed 100, while its weakest is Defense 78." |
| Competitive context | "A potent special attacker with Fire Blast... Weak to Rock, Water, and Electric." |

**Move descriptions** incorporate:

| Component | Example (Thunderbolt) |
|:---|:---|
| Name, type, category | "Thunderbolt is a Electric-type Special move in Generation 1." |
| Numeric stats | "It has 95 base power, 100% accuracy, and 15 PP." |
| Effect & meta-context | "A reliable special Electric-type attack with a 10% paralysis chance. One of the best coverage moves in Gen 1." |

The descriptions are intentionally verbose (~50–100 words each) to give the sentence encoder maximum semantic signal. The learned projection layer downstream will compress this to the 64 dimensions that matter for battle decisions.

### 1.4 Learned Projection

The frozen 768-dim embeddings are mapped into the RL latent space by learned linear projections:

```
Frozen LLM Embedding (768)  →  nn.Linear(768, 64)  →  Projected Embedding (64)
```

Separate projection heads are used for species and moves, allowing each to learn different aspects of the semantic space. A third projection handles the unknown token.

| Projection | Input → Output | Trainable | Purpose |
|:---|:---:|:---:|:---|
| `species_proj` | 768 → 64 | Yes | Map species semantics to RL space |
| `move_proj` | 768 → 64 | Yes | Map move semantics to RL space |
| `unknown_proj` | 768 → 64 | Yes | Map "unrevealed Pokémon" to RL space |
| `no_move_embedding` | — (64) | Yes | Learned vector for empty move slots |

### 1.5 Embedding Verification

After generation, cosine similarity checks confirm the embeddings capture meaningful relationships:

**Move Similarities** (high similarity = encoder understands they're related):

| Pair | Cosine Sim | Interpretation |
|:---|:---:|:---|
| Thunderbolt ↔ Thunder | +0.810 | Same-type variants correctly clustered |
| Sleep Powder ↔ Spore | +0.572 | Functional similarity (both sleep moves) |
| Surf ↔ Ice Beam | +0.434 | Common coverage pair recognized |
| Swords Dance ↔ Amnesia | +0.373 | Both +2 boosts, but different stat targets |
| Earthquake ↔ Psychic | +0.480 | Different types, moderate baseline similarity |

**Species Similarities:**

| Pair | Cosine Sim | Interpretation |
|:---|:---:|:---|
| Charizard ↔ Arcanine | +0.529 | Both Fire-type attackers |
| Alakazam ↔ Starmie | +0.537 | Both top-tier Psychic special attackers |
| Snorlax ↔ Chansey | +0.498 | Both bulky Normal-type walls |
| Charizard ↔ Geodude | +0.500 | Different roles, moderate similarity |
| Mewtwo ↔ Magikarp | +0.512 | Highest vs lowest BST, still moderate |

Species similarities cluster more tightly (0.49–0.54) than moves because all descriptions share the "Pokémon with stats" template. The learned projection amplifies battle-relevant dimensions during training.

---

## 2. Per-Pokémon Encoder

The per-Pokémon encoder combines projected entity embeddings with numeric battle-state features and produces a fixed-size vector for each Pokémon slot. **Weights are shared** across all 12 Pokémon positions.

### 2.1 Per-Pokémon Feature Representation

Each Pokémon is represented by a **482-dimensional** feature vector composed of embedding features (320 dims) and numeric features (162 dims).

#### Embedding Features (320 dims)

| Feature | Dims | Source | Notes |
|:---|:---:|:---|:---|
| Species embedding | 64 | `species_proj(frozen_emb[species_idx])` | -1 index → unknown token |
| Move 1 embedding | 64 | `move_proj(frozen_emb[move1_idx])` | -1 index → learned `no_move` |
| Move 2 embedding | 64 | `move_proj(frozen_emb[move2_idx])` | |
| Move 3 embedding | 64 | `move_proj(frozen_emb[move3_idx])` | |
| Move 4 embedding | 64 | `move_proj(frozen_emb[move4_idx])` | |
| **Subtotal** | **320** | | |

#### Numeric Features (162 dims)

| Feature | Dims | Domain | Notes |
|:---|:---:|:---|:---|
| Current HP fraction | 7 | {0,1} | One-hot: 0% + 6 equal bins (following Wang) |
| Attack boost | 13 | {0,1} | One-hot: -6 to +6 |
| Defense boost | 13 | {0,1} | One-hot: -6 to +6 |
| Special boost | 13 | {0,1} | One-hot: -6 to +6. *Gen 1 has one Special stat* |
| Speed boost | 13 | {0,1} | One-hot: -6 to +6 |
| Accuracy boost | 13 | {0,1} | One-hot: -6 to +6 |
| Evasion boost | 13 | {0,1} | One-hot: -6 to +6 |
| Status condition | 8 | {0,1} | One-hot: none/sleep/poison/burn/paralysis/freeze/fainted/toxic |
| Toxic counter | 16 | {0,1} | One-hot: 0–15 turns of toxic damage |
| Sleep counter | 8 | {0,1} | One-hot: 0–7 turns asleep |
| Move 1 PP bin | 4 | {0,1} | ⌊PP^(1/3)⌋ binning (following Wang) |
| Move 2 PP bin | 4 | {0,1} | |
| Move 3 PP bin | 4 | {0,1} | |
| Move 4 PP bin | 4 | {0,1} | |
| Type(s) | 15 | {0,1} | Multi-hot: 15 Gen 1 types (1–2 active) |
| is_active | 1 | {0,1} | Currently on the field |
| is_opponent | 1 | {0,1} | Belongs to the opposing team |
| is_unknown | 1 | {0,1} | Unrevealed; all other features zeroed |
| first_turn | 1 | {0,1} | Just switched in this turn |
| must_recharge | 1 | {0,1} | Locked into recharge (e.g. after Hyper Beam) |
| is_fainted | 1 | {0,1} | HP = 0 |
| Confused | 1 | {0,1} | Volatile: confusion |
| Seeded | 1 | {0,1} | Volatile: Leech Seed |
| Partially trapped | 1 | {0,1} | Volatile: Wrap/Bind/Clamp/Fire Spin |
| Substitute active | 1 | {0,1} | Volatile: behind a Substitute |
| Focus Energy | 1 | {0,1} | Volatile: Focus Energy active (bugged in Gen 1) |
| Mist active | 1 | {0,1} | Volatile: Mist |
| Light Screen | 1 | {0,1} | Volatile: Light Screen (per-Pokémon in Gen 1) |
| Reflect | 1 | {0,1} | Volatile: Reflect (per-Pokémon in Gen 1) |
| **Subtotal** | **162** | | |

> **Comparison with Wang (Gen 4):** Wang's per-Pokémon representation uses 300 dimensions, including ability, item, 18 types, 38 volatile effects, and Gen 4-specific fields (Encore, Taunt, Magnet Rise, Slow Start, Protect counter, gender, weight/height). Our Gen 1 representation is leaner at 162 numeric dims because Gen 1 lacks abilities, items, and many of these volatile effects. However, we add 320 dims of semantic embedding features that Wang's architecture does not have.

### 2.2 Battle Context Representation

A single battle context token captures global state not tied to any individual Pokémon.

| Feature | Dims | Domain | Notes |
|:---|:---:|:---|:---|
| Weather | 5 | {0,1} | One-hot: none/sun/rain/sandstorm/hail |
| Reflect (our side) | 1 | {0,1} | Side-wide screen active |
| Light Screen (our side) | 1 | {0,1} | |
| Reflect (opponent side) | 1 | {0,1} | |
| Light Screen (opponent side) | 1 | {0,1} | |
| Turn count bin | 6 | {0,1} | One-hot: 0–4, 5–9, 10–19, 20–29, 30–39, 40+ |
| Force switch | 2 | {0,1} | One-hot: yes/no (e.g. after fainting) |
| # Unknown opponent Pokémon | 7 | {0,1} | One-hot: 0–6 remaining unrevealed |
| **Total** | **24** | | |

> **Gen 1 notes:** Weather is extremely rare in Gen 1 random battles (no weather-summoning abilities exist, and weather moves like Sandstorm don't exist yet). The encoding is kept for completeness. Entry hazards (Stealth Rock, Spikes, Toxic Spikes) do not exist in Gen 1 and are omitted entirely. Trick Room and Terrain also do not exist.

### 2.3 Encoding Pipeline

The per-Pokémon encoder follows a three-stage pipeline:

```
                    PER-POKÉMON ENCODER (shared weights, applied ×12)
                    ═══════════════════════════════════════════════

Stage 1: Entity Embedding Lookup + Projection
──────────────────────────────────────────────
    species_idx → frozen_emb[idx] (768) → species_proj → (64)
    move1_idx   → frozen_emb[idx] (768) → move_proj    → (64)
    move2_idx   → frozen_emb[idx] (768) → move_proj    → (64)
    move3_idx   → frozen_emb[idx] (768) → move_proj    → (64)
    move4_idx   → frozen_emb[idx] (768) → move_proj    → (64)
                                                               ────
                                                    Subtotal:  320 dims

Stage 2: Concatenation
──────────────────────
    [species_emb (64)] ⊕ [move_embs (256)] ⊕ [numeric_features (162)]
                            │
                            ▼
                      482-dim vector

Stage 3: Shared MLP
───────────────────
    Linear(482, 512) → ReLU → Dropout(0.1)
    Linear(512, 256) → ReLU → Dropout(0.1)
    LayerNorm(256)
          │
          ▼
    256-dim Pokémon token (ready for Transformer)
```

The **battle context** follows a similar but smaller pipeline:

```
    context_features (24) → Linear(24, 256) → ReLU → Dropout
                          → Linear(256, 256) → ReLU → Dropout
                          → LayerNorm(256)
                                │
                                ▼
                    256-dim context token
```

### 2.4 Parameter Counts

| Component | Parameters | Trainable |
|:---|---:|:---:|
| Entity projections (species, move, unknown) | 147,712 | Yes |
| Per-Pokémon MLP + LayerNorm | 379,136 | Yes |
| Battle context MLP + LayerNorm | 72,704 | Yes |
| **Total trainable** | **~600K** | |
| Frozen LLM embeddings (buffers) | 234,240 | No |

> These counts cover only the encoder. The Transformer backbone and actor-critic heads (coming next) will add the remaining parameters toward the 5–10M total target.

---

## 3. Transformer Encoder
The Transformer encoder is the relational backbone of the model. It takes the 13 tokens from the per-Pokémon encoder and allows every token to attend to every other token through self-attention — enabling the model to reason about type matchups, team synergies, and threats across both sides of the battle.

### 3.1 Architecture Details
 
| Parameter | Value | Notes |
|:---|:---:|:---|
| Layers | 4 | Each layer = multi-head attention + feedforward |
| Attention heads | 4 | 64 dims per head (256 / 4) |
| Model dimension | 256 | Matches per-Pokémon encoder output |
| Feedforward dim | 1024 | 4 × d_model (standard ratio) |
| Activation | GELU | Smoother than ReLU, standard in modern Transformers |
| Normalization | Pre-norm | LayerNorm before attention/FFN (more stable for RL) |
| Dropout | 0.1 | Applied in attention and feedforward layers |
| Attention mask | None (full) | Bidirectional — every token attends to all others |
 
**Pre-norm vs. post-norm:** We use pre-norm Transformers (LayerNorm applied *before* attention and feedforward sublayers) because they produce more stable gradients during training. This is important for PPO, where large policy updates can destabilize the network.
 
```
For each of 4 layers:
    ┌───────────────────────────────┐
    │  x_norm = LayerNorm(x)        │
    │  x = x + MultiHeadAttn(       │
    │          x_norm, x_norm,      │  Pre-norm attention
    │          x_norm)              │
    │                               │
    │  x_norm = LayerNorm(x)        │
    │  x = x + FFN(x_norm)          │  Pre-norm feedforward
    │      (256 → 1024 → 256)       │
    └───────────────────────────────┘
```
 
### 3.2 Positional Embeddings
 
The 13 token positions carry structural meaning, so we use **learned positional embeddings** (not sinusoidal):
 
| Position | Slot | Meaning |
|:---:|:---|:---|
| 0 | Own active Pokémon | The Pokémon currently on the field — **actor head reads this** |
| 1–5 | Own bench Pokémon | Bench slots 1–5 (potential switch targets) |
| 6 | Opponent active Pokémon | The opponent's current Pokémon |
| 7–11 | Opponent bench | May be unknown (encoded with unknown token) |
| 12 | Battle context | Weather, side conditions, turn count |
 
With only 13 positions, learned embeddings add just 3,328 parameters. They allow the model to learn that positions 0–5 are "friendly" and 6–11 are "hostile" without us hard-coding that distinction.
 
### 3.3 Why Self-Attention Matters for Pokémon
 
Prior Pokémon RL work (Wang 2024, Chen & Lin 2018) used flat MLPs that see the entire state as a single concatenated vector. A Transformer's self-attention mechanism offers a structural advantage: each Pokémon token can *selectively attend* to the tokens most relevant to its current situation.
 
For example, when deciding whether Charizard should use Fire Blast:
- Charizard's token (position 0) can attend to the opponent's active Pokémon (position 6) to assess type matchup
- It can attend to its own bench (positions 1–5) to evaluate switch options
- It can attend to the context token (position 12) for weather conditions
- It can attend to revealed opponent bench Pokémon (positions 7–11) to anticipate switches
 
The attention pattern is *learned* — the model discovers which cross-entity comparisons are useful for battle decisions through PPO training.
 
---
 
## 4. Actor-Critic Output Heads
 
Both output heads read from the Transformer's contextualized token representations. They share the entire backbone (encoder + Transformer) and diverge only at the final MLPs.
 
### 4.1 Actor Head (Policy)
 
The actor produces a probability distribution over the 9 possible actions.
 
```
Transformer Output
       │
       ▼
Token 0 (active Pokémon)  ──→  Linear(256, 256)  ──→  ReLU  ──→  Linear(256, 9)
                                                                       │
                                                                 action masking
                                                                       │
                                                                       ▼
                                                              Categorical distribution
```
 
**Why token 0?** The active Pokémon is the one making the decision. Through 4 layers of self-attention, token 0 has already incorporated information from all 12 other tokens — it sees the full battle state from the perspective of the Pokémon that needs to act. Reading a single token (rather than pooling) preserves the position-specific focus that attention learned.
 
**Initialization:** The final linear layer uses orthogonal initialization with a small gain (0.01), following CleanRL best practices. This produces near-uniform initial action probabilities, which is important for PPO — a strongly opinionated initial policy leads to poor exploration.
 
### 4.2 Action Masking
 
Action masking is critical for Pokémon battles because the set of valid actions changes every turn. Following Wang (2024): invalid action logits are set to `-inf` before softmax, giving them exactly 0 probability.
 
| Action Index | Action | When Invalid |
|:---:|:---|:---|
| 0 | Use move 1 | Move has 0 PP, or forced to switch |
| 1 | Use move 2 | Move has 0 PP, disabled, or forced to switch |
| 2 | Use move 3 | Move has 0 PP, disabled, or forced to switch |
| 3 | Use move 4 | Move has 0 PP, disabled, or forced to switch |
| 4 | Switch to bench slot 1 | Pokémon fainted, or trapped by Wrap/Bind |
| 5 | Switch to bench slot 2 | Pokémon fainted, or trapped |
| 6 | Switch to bench slot 3 | Pokémon fainted, or trapped |
| 7 | Switch to bench slot 4 | Pokémon fainted, or trapped |
| 8 | Switch to bench slot 5 | Pokémon fainted, or trapped |
 
The mask is a boolean tensor provided by the environment (Partner A's observation encoder). The model applies it as: `logits.masked_fill(~mask, -inf)`.
 
**During PPO updates**, the same saved action masks are reapplied when recomputing log probabilities under the updated policy. This ensures the probability ratio `π_new(a|s) / π_old(a|s)` is computed only over the valid action space.
 
### 4.3 Critic Head (Value)
 
The critic estimates how favorable the current battle state is, as a scalar value.
 
```
Transformer Output
       │
       ▼
Mean pool across all 13 tokens  ──→  Linear(256, 256)  ──→  ReLU  ──→  Linear(256, 1)
                                                                            │
                                                                            ▼
                                                                    Scalar value V(s)
```
 
**Why mean pooling (not token 0)?** The value of a battle state depends on the *global* situation — both teams' health, composition, and positioning. Mean pooling gives the critic equal access to all information. By contrast, the actor reads only token 0 because the *policy* should be from the active Pokémon's perspective.
 
**Reward structure** (from project outline): The reward is +1 for a win, -1 for a loss, and 0 for all other turns. The critic learns to predict the expected discounted return from the current state, which PPO uses for advantage estimation via GAE. (Should update to use recommended structure or some variant from poke-env: https://poke-env.readthedocs.io/en/stable/examples/reinforcement_learning.html#reinforcement-learning).
 
### 4.4 Full Model Parameter Summary
 
| Component | Parameters | % of Total |
|:---|---:|---:|
| Entity projections (species, move, unknown) | ~148K | 3.7% |
| Per-Pokémon MLP + LayerNorm | ~379K | 9.6% |
| Battle context MLP + LayerNorm | ~73K | 1.8% |
| Positional encoding (buffer) | 0 | 0% |
| Transformer encoder (4 layers) | ~3,160K | 79.8% |
| Actor head | ~134K | 3.4% |
| Critic head | ~66K | 1.7% |
| **Total trainable** | **~3.96M** | **100%** |
| Frozen LLM embeddings (buffers) | 234K | — |
 
The Transformer dominates the parameter budget at ~80%, which is expected — the relational reasoning it provides is the core computational investment. The total of ~4M is appropriate for Gen 1's reduced complexity compared to the outline's 5–10M target for Gen 4. If needed during Week 3 experiments, we can scale to dim-512 (~14M params) or add layers.
 
---

## 5. Gen 1 Adaptations

This architecture was originally designed for Gen 4 (following Wang 2024). Below is a summary of all Gen 1-specific changes.

| Feature | Gen 4 (Wang / Outline) | Gen 1 (Ours) | Reason |
|:---|:---|:---|:---|
| Species count | 296 | 151 | Gen 1 dex |
| Move count | ~200 | 153 | Gen 1 move pool |
| Abilities | ~100, embedded | None | Abilities don't exist until Gen 3 |
| Items | ~40, embedded | None | Not used in `gen1randombattle` |
| Special stat | SpA + SpD (separate) | Single "Special" | Gen 1 uses one unified Special stat |
| Stat boosts | 7 stats × 13 | 6 stats × 13 | No separate SpA/SpD boosts |
| Types | 18 | 15 | Gen 1 has no Dark, Steel, or Fairy |
| Volatile statuses | 38 effects | 8 effects | Gen 1 has far fewer volatile conditions |
| Entry hazards | SR, Spikes, T-Spikes | None | Hazards don't exist until Gen 2+ |
| Weather | 5 types + permanent | 5 types (rare) | No weather abilities in Gen 1 |
| Trick Room / Terrain | Encoded | Omitted | Don't exist in Gen 1 |
| Protect counter | Up to 5 | Omitted | Protect doesn't exist in Gen 1 |
| Gender | 3-way one-hot | Omitted | Gender mechanics don't exist in Gen 1 |
| Weight / Height | Log-binned | Omitted | No weight-based moves in Gen 1 |
| Per-Pokémon dims | ~300 numeric | 162 numeric + 320 embedding | Leaner numeric, richer semantic |
| Total per-Pokémon | ~300 | 482 | Semantic embeddings are the difference |

---

## File Structure

```
Network/
├── generate_embeddings.py     # LLM embedding generation for all Gen 1 entities
├── pokemon_encoder.py         # Per-Pokémon encoder (EntityEmbedding + MLP)
├── battle_model.py            # Full model: Transformer encoder + actor-critic heads
├── embeddings/                # Generated embedding files (after running generate_embeddings.py)
│   ├── species_embeddings.pt  #   Dict: {name → tensor(768,)}
│   ├── species_matrix.pt      #   Tensor: (151, 768) — for nn.Embedding init
│   ├── move_embeddings.pt     #   Dict: {name → tensor(768,)}
│   ├── move_matrix.pt         #   Tensor: (153, 768) — for nn.Embedding init
│   ├── unknown_embedding.pt   #   Tensor: (768,) — for unrevealed Pokémon
│   ├── embedding_index.pt     #   Dict: name↔index mappings
│   └── metadata.json          #   Generation metadata (model, date, counts)
└── README.md                  # This file
```

---

## Quick Start

### 1. Generate Embeddings

```bash
pip install sentence-transformers torch
python generate_embeddings.py --verify
```

This produces 768-dim Sentence-BERT embeddings for all 151 species and 153 moves, with cosine similarity verification.

### 2. Run Encoder Unit Tests

```bash
python pokemon_encoder.py
```

Runs 6 tests: shape verification, batched forward pass, unknown Pokémon handling, gradient flow, output statistics, and integration with real embeddings (if generated).

### 3. Run Full Model Unit Tests
 
```bash
python battle_model.py
```
 
Runs 11 tests covering the complete pipeline: Transformer standalone, action masking correctness, critic head, full forward pass, PPO get_action_and_value (sampling and evaluation), value-only pass for GAE, gradient flow through all components, parameter counts, force-switch scenario, device handling, and real embedding integration.
 
### 4. Use in Training (Preview)
 
```python
from battle_model import PokemonAgent, TransformerConfig
from pokemon_encoder import Gen1Config
import torch
 
gen1_config = Gen1Config()
tf_config = TransformerConfig()
 
# Load real embeddings
species_matrix = torch.load("embeddings/species_matrix.pt", weights_only=True, map_location="cpu")
move_matrix = torch.load("embeddings/move_matrix.pt", weights_only=True, map_location="cpu")
unknown_emb = torch.load("embeddings/unknown_embedding.pt", weights_only=True, map_location="cpu")
 
# Build full model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = PokemonAgent(gen1_config, tf_config, species_matrix, move_matrix, unknown_emb).to(device)
 
# Rollout: sample actions from the policy
action, log_prob, entropy, value = agent.get_action_and_value(
    species_indices, move_indices, numeric_features, context_features, action_mask
)
 
# PPO update: evaluate saved actions under the current policy
_, new_log_prob, new_entropy, new_value = agent.get_action_and_value(
    species_indices, move_indices, numeric_features, context_features,
    action_mask, action=saved_actions
)
ratio = (new_log_prob - old_log_prob).exp()  # for PPO clipping
```

---

## References

- **Wang (2024)** — *Winning at Pokémon Random Battles Using Reinforcement Learning.* MIT MEng thesis. PPO self-play + MCTS for gen4randombattles. Peaked rank 8 (1693 Elo). Our state representation tables are modeled after Wang's Appendix A. [[PDF]](https://dspace.mit.edu/handle/1721.1/156650)
- **Chen & Lin (2018)** — *Gotta Train 'Em All.* Stanford CS230. PPO + Node2Vec embeddings for Pokémon Showdown. [[PDF]](https://cs230.stanford.edu/projects_fall_2018/reports/12443578.pdf)
- **Tse (2019)** — *Learning Competitive Pokemon through Neural Network and Reinforcement Learning.* Stanford CS230. Supervised pretraining + Deep Q-Learning. [[PDF]](https://cs230.stanford.edu/projects_spring_2019/reports/18681282.pdf)
- **Reimers & Gurevych (2019)** — *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* The sentence encoder used for our entity embeddings. [[Paper]](https://arxiv.org/abs/1908.10084)