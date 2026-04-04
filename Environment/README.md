# Environment

All code pertaining to the environment for Pokemon battles (Poke-env) will live in here.

Okay so the battle object contains all of the state information. Info is https://poke-env.readthedocs.io/en/stable/modules/battle.html

# Environment → Model Integration Guide 

## Overview

This document specifies exactly what tensors the model (`PokemonAgent` in `battle_model.py`) expects from the environment, and how to extract them from poke-env's `battle` object. Your job is to write an `embed_battle(battle)` function that converts a raw poke-env battle state into a dictionary of 5 tensors.

The model's forward signature is:

```python
logits, value = agent(
    species_indices,   # (batch, 12)     int
    move_indices,      # (batch, 12, 4)  int
    numeric_features,  # (batch, 12, 162) float32
    context_features,  # (batch, 24)     float32
    action_mask,       # (batch, 9)      bool
)
```

During rollout collection (single battle step), `batch = 1`. During PPO updates, observations are batched.

---

## Step 0: Load the Embedding Index

The `generate_embeddings.py` script produces an `embedding_index.pt` file containing name-to-integer mappings. Load this **once** at initialization — not every step.

```python
import torch

# Load once at agent/environment init
index = torch.load("embeddings/embedding_index.pt", weights_only=True)
species_to_idx = index["species_to_idx"]  # dict: {"bulbasaur": 0, ..., "mew": 150}
move_to_idx    = index["move_to_idx"]     # dict: {"tackle": 0, ..., "bide": 152}
```

These dictionaries map lowercase, no-space names to integer indices. poke-env uses the same naming convention, so names match directly — **no string cleaning needed**.

### Quick verification

```python
species_to_idx["charizard"]   # → 5
species_to_idx["pikachu"]     # → 24
species_to_idx["mewtwo"]      # → 149

move_to_idx["bodyslam"]       # → 3
move_to_idx["earthquake"]     # → 37
move_to_idx["thunderbolt"]    # → 61
```

---

## Input 1: `species_indices` — shape `(12,)` int

An integer index for each of the 12 Pokémon slots. Use `-1` for unknown/unrevealed Pokémon.

### Slot ordering convention (must be consistent every step!)

| Slot | Meaning | Source |
|------|---------|--------|
| 0 | Your active Pokémon | `battle.active_pokemon` |
| 1–5 | Your bench (sorted by team slot) | `battle.team` minus active |
| 6 | Opponent active Pokémon | `battle.opponent_active_pokemon` |
| 7–11 | Opponent revealed bench + unknowns | `battle.opponent_team` minus active, pad with -1 |

### Code

```python
import numpy as np

def get_species_indices(battle, species_to_idx):
    indices = np.full(12, -1, dtype=np.int64)

    # --- Your team (slots 0-5) ---
    # Active Pokémon always goes in slot 0
    active = battle.active_pokemon
    if active:
        indices[0] = species_to_idx.get(active.species, -1)

    # Bench Pokémon in slots 1-5, sorted by team key for determinism
    bench = [mon for key, mon in sorted(battle.team.items())
             if mon != active]
    for i, mon in enumerate(bench[:5]):
        indices[1 + i] = species_to_idx.get(mon.species, -1)

    # --- Opponent team (slots 6-11) ---
    opp_active = battle.opponent_active_pokemon
    if opp_active:
        indices[6] = species_to_idx.get(opp_active.species, -1)

    # Revealed opponent bench in slots 7-11
    opp_bench = [mon for key, mon in sorted(battle.opponent_team.items())
                 if mon != opp_active]
    for i, mon in enumerate(opp_bench[:5]):
        indices[7 + i] = species_to_idx.get(mon.species, -1)

    # Slots for unrevealed opponents remain -1 (set by np.full above)
    return indices
```

**Key point:** The model's `EntityEmbeddingLayer` handles `-1` by substituting a learned "unknown" embedding. You don't need to do anything special — just pass -1.

---

## Input 2: `move_indices` — shape `(12, 4)` int

For each Pokémon slot, the 4 move indices. Use `-1` for unknown or empty move slots.

### Code

```python
def get_move_indices(battle, move_to_idx):
    indices = np.full((12, 4), -1, dtype=np.int64)

    # Helper: extract up to 4 move indices for a Pokémon
    def pokemon_moves(pokemon):
        if pokemon is None:
            return [-1, -1, -1, -1]
        # Sort by move ID for deterministic ordering
        move_ids = sorted(pokemon.moves.keys())
        result = [move_to_idx.get(mid, -1) for mid in move_ids[:4]]
        # Pad to 4 slots
        while len(result) < 4:
            result.append(-1)
        return result

    # Your team
    active = battle.active_pokemon
    indices[0] = pokemon_moves(active)

    bench = [mon for key, mon in sorted(battle.team.items())
             if mon != active]
    for i, mon in enumerate(bench[:5]):
        indices[1 + i] = pokemon_moves(mon)

    # Opponent team (only moves you've SEEN are in pokemon.moves)
    opp_active = battle.opponent_active_pokemon
    indices[6] = pokemon_moves(opp_active)

    opp_bench = [mon for key, mon in sorted(battle.opponent_team.items())
                 if mon != opp_active]
    for i, mon in enumerate(opp_bench[:5]):
        indices[7 + i] = pokemon_moves(mon)

    return indices
```

**Important:** For opponent Pokémon, `pokemon.moves` only contains moves that the opponent has *used so far in the battle*. Unseen move slots stay as -1.  The model uses a learned "no move" embedding for these.

---

## Input 3: `numeric_features` — shape `(12, 162)` float32

This is the per-Pokémon numeric feature vector. Each Pokémon gets 162 features, broken down as follows:

| Feature | Dims | Encoding | poke-env Source |
|---------|------|----------|-----------------|
| HP bins | 7 | One-hot: bin 0 = fainted (0%), bins 1-6 = equal splits of (0%, 100%] | `pokemon.current_hp_fraction` |
| Status condition | 8 | One-hot: none/sleep/poison/burn/paralysis/freeze/fainted/toxic | `pokemon.status` |
| Toxic counter | 16 | One-hot: 0-15 turns | `pokemon.status_counter` (if toxic) |
| Sleep counter | 8 | One-hot: 0-7 turns | `pokemon.status_counter` (if sleep) |
| Stat boosts | 78 | 6 stats × 13-dim one-hot each (-6 to +6) | `pokemon.boosts` |
| Move PP bins | 16 | 4 moves × 4-dim one-hot each | `move.current_pp` |
| Types | 15 | Multi-hot over 15 Gen 1 types | `pokemon.types` |
| Binary flags | 6 | is_active, is_opponent, is_unknown, first_turn, must_recharge, is_fainted | various |
| Volatile statuses | 8 | confused/seeded/trapped/substitute/focus_energy/mist/light_screen/reflect | `pokemon.effects` |
| **Total** | **162** | | |

### Encoding details

**HP binning** — 7 bins:
```python
def hp_to_onehot(hp_fraction):
    """Convert HP fraction [0.0, 1.0] to 7-dim one-hot."""
    vec = np.zeros(7)
    if hp_fraction == 0.0:
        vec[0] = 1  # fainted
    else:
        # 6 equal bins: (0%, 16.7%], (16.7%, 33.3%], ..., (83.3%, 100%]
        bin_idx = min(int(hp_fraction * 6 - 1e-9), 5)  # 0-5
        vec[1 + bin_idx] = 1
    return vec
```

**PP binning** — cube-root formula, 4 bins per move:
```python
def pp_to_onehot(current_pp):
    """Convert PP count to 4-dim one-hot using cube-root binning."""
    vec = np.zeros(4)
    bin_idx = min(int(current_pp ** (1/3)), 3)  # floor(pp^(1/3)), capped at 3
    vec[bin_idx] = 1
    return vec
```

**Stat boosts** — 13-dim one-hot per stat:
```python
def boost_to_onehot(boost_value):
    """Convert boost stage (-6 to +6) to 13-dim one-hot."""
    vec = np.zeros(13)
    vec[boost_value + 6] = 1  # shift so -6 → index 0, +6 → index 12
    return vec
```

**Gen 1 boost stats from poke-env:** The `pokemon.boosts` dict has keys `atk`, `def`, `spa`, `spe`, `accuracy`, `evasion`. In Gen 1, `spa` covers the unified "Special" stat. Encode all 6 in this order: Attack, Defense, Special, Speed, Accuracy, Evasion.

**Type encoding** — 15 Gen 1 types:
```python
GEN1_TYPES = [
    "normal", "fighting", "flying", "poison", "ground",
    "rock", "bug", "ghost", "fire", "water",
    "grass", "electric", "psychic", "ice", "dragon"
]
TYPE_TO_IDX = {t: i for i, t in enumerate(GEN1_TYPES)}

def types_to_multihot(pokemon_types):
    """Convert Pokémon types to 15-dim multi-hot vector."""
    vec = np.zeros(15)
    for ptype in pokemon_types:
        if ptype is not None:
            type_name = ptype.name.lower()  # PokemonType enum → string
            if type_name in TYPE_TO_IDX:
                vec[TYPE_TO_IDX[type_name]] = 1
    return vec
```

**Unknown Pokémon:** For unrevealed opponent slots, set `is_unknown = 1` and all other numeric features to 0. The model handles this correctly.

---

## Input 4: `context_features` — shape `(24,)` float32

Battle-wide context information:

| Feature | Dims | Encoding | poke-env Source |
|---------|------|----------|-----------------|
| Weather | 5 | One-hot: none/sun/rain/sandstorm/hail | `battle.weather` |
| Side conditions | 4 | Binary: reflect_ours, light_screen_ours, reflect_theirs, light_screen_theirs | `battle.side_conditions`, `battle.opponent_side_conditions` |
| Turn count | 6 | One-hot bins: 0-4, 5-9, 10-19, 20-29, 30-39, 40+ | `battle.turn` |
| Force switch | 2 | One-hot: yes/no | Check if forced to switch |
| Unknown opponent count | 7 | One-hot: 0-6 unrevealed opponents | `6 - len(battle.opponent_team)` |
| **Total** | **24** | | |

### Turn count binning

```python
def turn_to_onehot(turn):
    vec = np.zeros(6)
    if turn <= 4:     vec[0] = 1
    elif turn <= 9:   vec[1] = 1
    elif turn <= 19:  vec[2] = 1
    elif turn <= 29:  vec[3] = 1
    elif turn <= 39:  vec[4] = 1
    else:             vec[5] = 1
    return vec
```

---

## Input 5: `action_mask` — shape `(9,)` bool

The action space has 9 slots. `True` = legal action, `False` = illegal (will be masked to -inf logits).

| Action Index | Meaning | Legal when... |
|-------------|---------|---------------|
| 0 | Use move in slot 0 | Move exists and is available this turn |
| 1 | Use move in slot 1 | Move exists and is available this turn |
| 2 | Use move in slot 2 | Move exists and is available this turn |
| 3 | Use move in slot 3 | Move exists and is available this turn |
| 4 | Switch to bench slot 1 | Pokémon is alive and not currently active |
| 5 | Switch to bench slot 2 | Pokémon is alive and not currently active |
| 6 | Switch to bench slot 3 | Pokémon is alive and not currently active |
| 7 | Switch to bench slot 4 | Pokémon is alive and not currently active |
| 8 | Switch to bench slot 5 | Pokémon is alive and not currently active |

### Code

```python
def get_action_mask(battle):
    mask = np.zeros(9, dtype=bool)

    # Moves 0-3: must match the SAME ordering used in move_indices for slot 0
    available_move_ids = sorted([m.id for m in battle.available_moves])
    active_move_ids = sorted(battle.active_pokemon.moves.keys()) if battle.active_pokemon else []

    for move_id in available_move_ids:
        if move_id in active_move_ids:
            slot_idx = active_move_ids.index(move_id)
            if slot_idx < 4:
                mask[slot_idx] = True

    # Switches 4-8: must match the SAME bench ordering used in species_indices slots 1-5
    bench = [mon for key, mon in sorted(battle.team.items())
             if mon != battle.active_pokemon]
    available_switch_species = {mon.species for mon in battle.available_switches}

    for i, mon in enumerate(bench[:5]):
        if mon.species in available_switch_species:
            mask[4 + i] = True

    return mask
```

**Critical:** The move ordering in the mask MUST match the move ordering in `move_indices[0]` (your active Pokémon's moves). Same for switch ordering matching `species_indices[1:6]`. If these drift, the model will select action 2 thinking it's Earthquake but the environment will execute Thunderbolt.

---

## Action → Battle Order Conversion

After the model samples an action (0–8), convert it back to a poke-env order:

```python
def action_to_order(action_idx, battle):
    # Get consistently-ordered moves and bench (same ordering as encoding!)
    active_move_ids = sorted(battle.active_pokemon.moves.keys())
    bench = [mon for key, mon in sorted(battle.team.items())
             if mon != battle.active_pokemon]

    if action_idx < 4:
        # It's a move
        move_id = active_move_ids[action_idx]
        move = battle.active_pokemon.moves[move_id]
        return battle.create_order(move)
    else:
        # It's a switch
        bench_idx = action_idx - 4
        target = bench[bench_idx]
        return battle.create_order(target)
```

---

## Reward Function

### Phase 1 (pipeline debugging, < 1M steps) — use shaped rewards:

```python
def calc_reward(self, battle) -> float:
    return self.reward_computing_helper(
        battle,
        fainted_value=1.0,
        hp_value=0.5,
        status_value=0.0,
        victory_value=30.0,
    )
```

### Phase 2 (main training, 5M+ steps) — switch to sparse rewards:

```python
def calc_reward(self, battle) -> float:
    if battle.won:
        return 1.0
    elif battle.lost:
        return -1.0
    return 0.0
```

Use γ = 0.9999 with sparse rewards (signal must propagate ~30+ turns). Use γ = 0.99 with shaped rewards.

---

## Complete `embed_battle` Skeleton

```python
def embed_battle(battle, species_to_idx, move_to_idx):
    """
    Convert a poke-env battle object into model-ready tensors.
    
    Returns dict with keys:
        species_indices  - (12,)     int64
        move_indices     - (12, 4)   int64
        numeric_features - (12, 162) float32
        context_features - (24,)     float32
        action_mask      - (9,)      bool
    """
    species_indices  = get_species_indices(battle, species_to_idx)
    move_indices     = get_move_indices(battle, move_to_idx)
    numeric_features = get_numeric_features(battle)      # you implement this
    context_features = get_context_features(battle)      # you implement this
    action_mask      = get_action_mask(battle)

    return {
        "species_indices":  torch.tensor(species_indices).unsqueeze(0),   # (1, 12)
        "move_indices":     torch.tensor(move_indices).unsqueeze(0),      # (1, 12, 4)
        "numeric_features": torch.tensor(numeric_features).unsqueeze(0),  # (1, 12, 162)
        "context_features": torch.tensor(context_features).unsqueeze(0),  # (1, 24)
        "action_mask":      torch.tensor(action_mask).unsqueeze(0),       # (1, 9)
    }
```

---

## Consistency Checklist

Before integration, verify these invariants:

- [ ] Bench Pokémon always sorted by `sorted(battle.team.items())` key
- [ ] Moves within each Pokémon always sorted by `sorted(pokemon.moves.keys())`
- [ ] Opponent bench sorted same way, unknowns always at the end (slots 9-11)
- [ ] Action mask move ordering matches `move_indices[0]` ordering exactly
- [ ] Action mask switch ordering matches `species_indices[1:6]` ordering exactly  
- [ ] Action-to-order conversion uses the **same** sorting as encoding
- [ ] Unknown Pokémon use `-1` for species and move indices, `is_unknown=1` flag, all other features 0
- [ ] `embed_battle()` returns the `.unsqueeze(0)` batch dimension

If any of these are wrong, the model will learn a broken policy. The most common bug is action indices drifting when Pokémon faint — **never collapse/shift the slot assignments**. If bench slot 3's Pokémon faints, action 6 becomes masked `False`, but action 7 still maps to bench slot 4 (not slot 3).

---

## Files You Need

From Partner B's side:

| File | What it provides |
|------|-----------------|
| `embeddings/embedding_index.pt` | `species_to_idx` and `move_to_idx` dictionaries |
| `pokemon_encoder.py` | `Gen1Config` with all dimension constants |
| `battle_model.py` | `PokemonAgent` model with `forward()` and `get_action_and_value()` |
| `generate_embeddings.py` | Run this first to produce the embeddings directory |