"""
env.py
======
poke-env Player implementation for the GEN 1 PPO agent

This module bridges the poke-env battle interface and our PokemonAgent
network. It is responsible for:

    1. embed_battle(battle)  -> dict of model-input tensors
    2. choose_move(battle)   -> samples an action from the policy and
                                 stores the (obs, actions, log_prob, value,
                                 mask) transition for later PPO updates
    3. action_to_order(...)  -> converts a model action index back to a
                                 poke-env BattleOrder
    4. calc_reward(battle)   -> Sparse reward: +1 win / -1 loss / 0 otherwise
    5. on_battle_end(battle) -> finalizes the trajectory: assigns the terminal
                                reward to the last stored transition and marks
                                done = True

Reward design (decided up front for all of training):
    Spase: +1 win, -1 loss, 0 every other turn
    Paried with gamma = 0.99 in the PPO trainer. The critic + GAE handle
    credit assignment over the ~20-40 turn horizon of a Gen 1 random battle.
    Rationale: optimizes the true objective (winning), avoids reward-hacking,
    matches Wang's paper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from poke_env.player import Player
import numpy as np
import torch

# ================================
# Constants
# ================================

NUM_POKEMON_SLOTS = 12       # 6 own + 6 opponent
NUM_MOVES_PER_POKEMON = 4
NUM_NUMERIC_FEATURES = 162   # per-Pokemon numeric vector size
NUM_CONTEXT_FEATURES = 24    # battle context vector size
NUM_ACTIONS = 9              # 4 moves + 5 switches

# Status -> one-hot index
STATUS_NONE = 0
NUM_STATUS_SLOTS = 8
STATUS_TO_IDX = {"slp": 1, "psn": 2, "brn": 3, "par": 4, "frz": 5, "fnt": 6, "tox": 7}

# Type -> one-hot index
GEN1_TYPES = [
    "normal", "fighting", "flying", "poison", "ground", "rock", "bug",
    "ghost", "fire", "water", "grass", "electric", "psychic", "ice", "dragon"
]
TYPE_TO_IDX = {t: i for i, t in enumerate(GEN1_TYPES)}
NUM_TYPES = len(GEN1_TYPES)

# Boosts
BOOST_KEYS = ["atk", "def", "spa", "spe", "accuracy", "evasion"]
NUM_BOOST_STATS = 6
BOOST_LEVELS = 13 # -6..+6

# HP / PP / Counter widths
HP_BINS = 7
TOXIC_COUNTER_BINS = 16
SLEEP_COUNTER_BINS = 8
PP_BINS = 4

# Slot offsets
OFF_HP        = 0                                # +7  -> 7
OFF_STATUS    = OFF_HP + HP_BINS                 # +8  -> 15
OFF_TOX_CNT   = OFF_STATUS + NUM_STATUS_SLOTS    # +16 -> 31
OFF_SLP_CNT   = OFF_TOX_CNT + TOXIC_COUNTER_BINS # +8  -> 39
OFF_BOOSTS    = OFF_SLP_CNT + SLEEP_COUNTER_BINS # +78 -> 117
OFF_PP        = OFF_BOOSTS + NUM_BOOST_STATS * BOOST_LEVELS # +16 -> 133
OFF_TYPES     = OFF_PP + NUM_MOVES_PER_POKEMON * PP_BINS    # +15 -> 148
OFF_FLAGS     = OFF_TYPES + NUM_TYPES            # +6 -> 154
OFF_VOLATILE  = OFF_FLAGS + 6                    # +8 -> 162
TOTAL_NUMERIC = OFF_VOLATILE + 8                 # = 162

# Sanity check
assert TOTAL_NUMERIC == NUM_NUMERIC_FEATURES, (
    f"Numeric layout mismatch: {TOTAL_NUMERIC} != {NUM_NUMERIC_FEATURES}"
)

# Binary flag indices
FLAG_IS_ACTIVE     = 0
FLAG_IS_OPPONENT   = 1
FLAG_IS_UNKNOWN    = 2
FLAG_FIRST_TURN    = 3
FLAG_MUST_RECHARGE = 4
FLAG_IS_FAINTED    = 5

# Volatile flag indices
VOL_CONFUSED       = 0
VOL_SEEDED         = 1
VOL_PARTIALLY_TRAP = 2
VOL_SUBSTITUTE     = 3
VOL_FOCUS_ENERGY   = 4
VOL_MIST           = 5
VOL_LIGHT_SCREEN   = 6
VOL_REFLECT        = 7

# poke-env effect enum -> volatile slot
VOLATILE_NAME_TO_SLOT = {
    "confusion": VOL_CONFUSED, "leechseed": VOL_SEEDED, "wrap": VOL_PARTIALLY_TRAP,
    "bind": VOL_PARTIALLY_TRAP, "clamp": VOL_PARTIALLY_TRAP, "firespin": VOL_PARTIALLY_TRAP,
    "partiallytrapped": VOL_PARTIALLY_TRAP, "substitute": VOL_SUBSTITUTE, "focusenergy": VOL_FOCUS_ENERGY,
    "mist": VOL_MIST, "lightscreen": VOL_LIGHT_SCREEN, "reflect": VOL_REFLECT
}

# ================================
# String normalization helpers
# ================================

def _norm(x: Any) -> str:
    if x is None:
        return ""
    s = x.name if hasattr(x, "name") else str(x)
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")

# ===============================
# Atomic Encoders
# ===============================

def _hp_onehot(hp_fraction: float, out: np.ndarray) -> None:
    """7 bins: 0 = fainted, 1..6 equal splits of (0%, 100%]"""
    if hp_fraction is None or hp_fraction <= 0.0:
        out[OFF_HP + 0] = 1.0
        return
    bin_idx = min(int(hp_fraction * 6 - 1e-9), 5)
    out[OFF_HP + 1 + bin_idx] = 1.0

def _status_onehot(mon, out: np.ndarray) -> None:
    """8 status slots. Defaults to 'none' if no status set"""
    if mon is None:
        out[OFF_STATUS + STATUS_NONE] = 1.0
        return
    
    # Fainted takes precedence
    if getattr(mon, "fainted", False):
        out[OFF_STATUS + STATUS_FAINTED] = 1.0
        return
    
    status = getattr(mon, "status", None)
    if status is None:
        out[OFF_STATUS + STATUS_NONE] = 1.0
        return
    
    key = _norm(status)
    idx = STATUS_TO_IDX.get(key, None)
    if idx is None:
        out[OFF_STATUS + STATUS_NONE] = 1.0
    else:
        out[OFF_STATUS + idx] = 1.0

def _toxic_counter_onehot(mon, out: np.ndarray) -> None:
    """16 bins. Active only when status is 'tox'."""
    if mon is None:
        out[OFF_TOX_CNT + 0] = 1.0
        return
    is_toxic = (_norm(getattr(mon, "status", None)) == "tox")
    if not is_toxic:
        out[OFF_TOX_CNT + 0] = 1.0
        return
    counter = int(getattr(mon, "status_counter", 0) or 0)
    bin_idx = max(0, min(counter, TOXIC_COUNTER_BINS - 1))
    out[OFF_TOX_CNT + bin_idx] = 1.0

def _sleep_counter_onehot(mon, out: np.ndarray) -> None:
    """8 bins. Active only when status is 'slp'."""
    if mon is None:
        out[OFF_SLP_CNT + 0] = 1.0
        return
    is_sleep = (_norm(getattr(mon, "status", None)) == "slp")
    if not is_sleep:
        out[OFF_SLP_CNT + 0] = 1.0
        return
    counter = int(getattr(mon, "status_counter", 0) or 0)
    bin_idx = max(0, min(counter, SLEEP_COUNTER_BINS - 1))
    out[OFF_SLP_CNT + bin_idx] = 1.0

def _boosts_onehot(mon, out: np.darray) -> None:
    """6 stats * 13-dim one-hot in BOOST_KEYS order. Default boost = 0."""
    boosts = (getattr(mon, "boosts", None) or {}) if mon is not None else {}
    for i, key in enumerate(BOOST_KEYS):
        raw = int(boosts.get(key, 0) or 0)
        raw = max(-6, min(6, raw))
        one_hot_idx = raw + 6
        out[OFF_BOOSTS + i * BOOST_LEVELS + one_hot_idx] = 1.0

def _pp_onehot(mon, out: np.ndarray) -> None:
    """
    4 moves * 4-dim one-hot. Bin = floor(current_pp ** (1/3)), capped at 3.
    Move ordering: sorted(mon.moves.key()), matching get_move_indices().

    For Pokemon with fewer than 4 moves, missing slots get bin 0 (= "no PP").
    This is a slight abust of the bin: bin 0 also covers PP=0 for an existing
    move. The model can disambiguate via the move embedding (-1 for empty slot).
    """
    if mon is None:
        for i in range(NUM_MOVES_PER_POKEMON):
            out[OFF_PP + i * PP_BINS + 0] = 1.0
        return
    
    sorted_ids = sorted((getattr(mon, "moves", None) or {}).keys())
    for i in range(NUM_MOVES_PER_POKEMON):
        if i < len(sorted_ids):
            move = mon.moves[sorted_ids[i]]
            cur_pp = int(getattr(move, "current_pp", 0) or 0)
            bin_idx = min(int(cur_pp ** (1.0 / 3.0)), PP_BINS - 1)
        else:
            bin_idx = 0
        out[OFF_PP + i * PP_BINS + bin_idx] = 1.0

def _types_multihot(mon, out: np.ndarray) -> None:
    """15-dim multi-hot over Gen 1 types"""
    if mon is None:
        return
    try:
        types = [t for t in (getattr(mon, "types", None) or []) if t is not None]
    except Exception:
        types = []
    for t in types:
        key = _norm(t)
        if key in TYPE_TO_IDX:
            out[OFF_TYPES + TYPE_TO_IDX[key]] = 1.0

def _binary_flags(mon, is_active: bool, is_opponent: bool, is_unknown: bool, out: np.ndarray) -> None:
    """6 binary flags"""
    base = OFF_FLAGS
    out[base + FLAG_IS_ACTIVE]   = 1.0 if (is_active and not is_unknown) else 0.0
    out[base + FLAG_IS_OPPONENT] = 1.0 if is_opponent else 0.0
    out[base + FLAG_IS_UNKNOWN]  = 1.0 if is_unknown else 0.0

    if mon is None or is_unknown:
        return
    
    first_turn = bool(getattr(mon, "first_turn", False))
    out[base + FLAG_FIRST_TURN] = 1.0 if first_turn else 0.0

    must_recharge = bool(getattr(mon, "must_recharge", False))
    out[base + FLAG_MUST_RECHARGE] = 1.0 if must_recharge else 0.0

    out[base + FLAG_IS_FAINTED] = 1.0 if bool(getattr(mon, "fainted", False)) else 0.0

def _volatile_flags(mon, out: np.ndarray) -> None:
    """
    8 binary flags for volatile statuses

    poke-env stores active effects in 'pokemon.effects', which is a dict
    {Effect: counter}. We normalize each effect's name and look it up in
    VOLATILE_NAME_TO_SLOT
    """
    if mon is None:
        return
    effects = getattr(mon, "effects", None) or {}
    for effect in effects.keys():
        key = _norm(effect)
        slot = VOLATILE_NAME_TO_SLOT.get(key)
        if slot is not None:
            out[OFF_VOLATILE + slot] = 1.0

# ==============================
# Per-Pokemon assembly
# ==============================

def _pokemon_numeric_features(mon, is_active: bool, is_opponent: bool, is_unknown: bool) -> np.ndarray:
    """
    Build the (162,) numeric feature vector for a single Pokémon slot

    For unknown slots: only is_unknown and is_opponent flags are set; everything
    else is zero
    """
    features = np.zeros(NUM_NUMERIC_FEATURES, dtype=np.float32)

    if is_unknown or mon is None:
        features[OFF_FLAGS + FLAG_IS_UNKNOWN] = 1.0
        if is_opponent:
            features[OFF_FLAGS + FLAG_IS_OPPONENT] = 1.0
        return features
    
    # All atomic encoders
    _hp_onehot(float(getattr(mon, "current_hp_fraction", 0.0) or 0.0), features)
    _status_onehot(mon, features)
    _toxic_counter_onehot(mon, features)
    _sleep_counter_onehot(mon, features)
    _boosts_onehot(mon, features)
    _pp_onehot(mon, features)
    _types_multihot(mon, features)
    _binary_flags(mon, is_active, is_opponent, is_unknown, features)
    _volatile_flags(mon, features)

    return features

# ==============================
# Battle context
# ==============================

# Context offsets
CTX_OFF_WEATHER     = 0  # + 5
CTX_OFF_SCREENS     = 5  # + 4 (reflect_us, ls_us, reflect_them, ls_them)
CTX_OFF_TURN        = 9  # + 6
CTX_OFF_FORCE_SW    = 15 # +2
CTX_OFF_UNKNOWN_OPP = 17 # +7
CTX_TOTAL           = 24
assert CTX_OFF_UNKNOWN_OPP + 7 == CTX_TOTAL == NUM_CONTEXT_FEATURES

WEATHER_TO_IDX = {"none": 0, "sun": 1, "rain": 2, "sandstorm": 3, "hail": 4,
                  "sunnyday": 1, "raindance": 2, "sand": 3}

def _context_features(battle) -> np.ndarray:
    features = np.zeros(NUM_CONTEXT_FEATURES, dtype=np.float32)

    # Weather
    weather = battle.weather
    if weather is None or (isinstance(weather, dict) and len(weather) == 0):
        features[CTX_OFF_WEATHER + 0] = 1.0
    else:
        # battle.weather can be a dict {Weather: turn} or a single enum
        if isinstance(weather, dict):
            key = _norm(next(iter(weather.keys())))
        else:
            key = _norm(weather)
        features[CTX_OFF_WEATHER + WEATHER_TO_IDX.get(key, 0)] = 1.0

    # Side conditions: Reflect / Light Screen each side
    def _has(side_conds, name: str) -> bool:
        if not side_conds:
            return False
        for cond in side_conds.keys():
            if _norm(cond) == name:
                return True
        return False
    
    features[CTX_OFF_SCREENS + 0] = 1.0 if _has(battle.side_conditions, "reflect") else 0.0
    features[CTX_OFF_SCREENS + 1] = 1.0 if _has(battle.side_conditions, "lightscreen") else 0.0
    features[CTX_OFF_SCREENS + 2] = 1.0 if _has(battle.opponent_side_conditions, "reflect") else 0.0
    features[CTX_OFF_SCREENS + 3] = 1.0 if _has(battle.opponent_side_conditions, "lightscreen") else 0.0

    # Turn count bin
    turn = int(battle.turn or 0)
    if   turn <= 4:  bin_idx = 0
    elif turn <= 9:  bin_idx = 1
    elif turn <= 19: bin_idx = 2
    elif turn <= 29: bin_idx = 3
    elif turn <= 39: bin_idx = 4
    else:            bin_idx = 5
    features[CTX_OFF_TURN + bin_idx] = 1.0

    # Force switch
    force_sw = bool(getattr(battle, "force_switch", False))
    if isinstance(force_sw, list):
        force_sw = any(force_sw)
    features[CTX_OFF_FORCE_SW + (1 if force_sw else 0)] = 1.0

    # Unknown opponent count
    num_known_opp = len(battle.opponent_team or {})
    num_unknown = max(0, min(6, 6 - num_known_opp))
    features[CTX_OFF_UNKNOWN_OPP + num_unknown] = 1.0

    return features

# =============================
# Slot ordering helpers
# =============================

def _ordered_own_team(battle) -> List:
    """Active first, then bench in deterministic sorted order. Length 6 (None-padded)."""
    active = battle.active_pokemon
    bench = [mon for key, mon in sorted(battle.team.items()) if mon is not active]
    result = ([active] if active is not None else [None]) + bench
    while len(result) < 6:
        result.append(None)
    return result[:6]

def _ordered_opp_team(battle) -> List:
    """Opponent active first, then known bench. None-padded for unrevealed slots."""
    active = battle.opponent_active_pokemon
    bench = [mon for key, mon in sorted(battle.opponent_team.items()) if mon is not active]
    result = ([active] if active is not None else [None]) + bench
    while len(result) < 6:
        result.append(None)
    return result[:6]

# ====================================
# Top-level embed_battle
# ====================================

def embed_battle(battle, species_to_idx: Dict[str, int],
                 move_to_idx: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """
    Convert a poke-env battle into the dict of tensors PokemonAgent expects.

    All tensors include a leading batch dim of size 1

    Returns:
        species_indices  : (1, 12)      int64
        move_indices     : (1, 12, 4)   int64
        numeric_features : (1, 12, 162) float32
        context_features : (1, 24)      float32
        action_mask      : (1, 9)       bool
    """
    own = _ordered_own_team(battle)
    opp = _ordered_opp_team

    species_indices = np.full(NUM_POKEMON_SLOTS, -1, dtype=np.int64)
    move_indices = np.full((NUM_POKEMON_SLOTS, NUM_MOVES_PER_POKEMON), -1, dtype=np.int64)
    numeric = np.zeros((NUM_POKEMON_SLOTS, NUM_NUMERIC_FEATURES), dtype=np.float32)

    # Slots 0..5 = own team
    for i, mon in enumerate(own):
        slot = i # 0..5
        is_active = (i == 0)
        is_unknown = (mon is None)
        if not is_unknown:
            species_indices[slot] = species_to_idx.get(_norm(mon.species), -1)
            sorted_move_ids = sorted((getattr(mon, "moves", None) or {}).keys())
            for j, mid in enumerate(sorted_move_ids[:NUM_MOVES_PER_POKEMON]):
                move_indices[slot, j] = move_to_idx.get(_norm(mid), -1)
        numeric_slot = _pokemon_numeric_features(
            mon, is_active=is_active, is_opponent=False, is_unknown=is_unknown
        )

    # Slots 6..11 = opponent team
    for i, mon in enumerate(opp):
        slot = 6 + i
        is_active = (i == 0)
        is_unknown = (mon is None)
        if not is_unknown:
            species_indices[slot] = species_to_idx(_norm(mon.species), -1)
            sorted_move_ids = sorted((getattr(mon, "moves", None) or {}).keys())
            for j, mid in enumerate(sorted_move_ids[:NUM_MOVES_PER_POKEMON]):
                move_indices[slot, j] = move_to_idx.get(_norm(mid), -1)
        numeric_slot = _pokemon_numeric_features(
            mon, is_active=is_active, is_opponent=True, is_unknown=is_unknown
        )

    context = _context_features(battle)
    mask = get_action_mask(battle)

    return {
        "species_indices":  torch.from_numpy(species_indices).unsqueeze(0),
        "move_indices":     torch.from_numpy(move_indices).unsqueeze(0),
        "numeric_features": torch.from_numpy(numeric).unsqueeze(0),
        "context_features": torch.from_numpy(context).unsqueeze(0),
        "action_mask":      torch.from_numpy(mask).unsqueeze(0),
    }

# =============================================
# Action mask + action -> order conversion
# =============================================

def get_action_mask(battle) -> np.ndarray:
    """
    (9, bool). True = legal this turn

    Action layout:
        0..3 : moves (sorted by move id, matching move_indices for the active mon)
        4..8 : switches to bench slots 1..5 (sorted by team key, matching species_indices[1:6])
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)

    active = battle.active_pokemon
    if active is not None:
        sorted_move_ids = sorted((getattr(active, "moves", None) or {}).keys())
        available_ids = {m.id for m in (battle.available_moves or [])}
        for i, mid in enumerate(sorted_move_ids[:NUM_MOVES_PER_POKEMON]):
            if mid in available_ids:
                mask[i] = True

    bench = [mon for key, mon in sorted(battle.team.items()) if mon is not active]
    available_switch_species = {m.species for m in (battle.available_switches or [])}
    for i, mon in enumerate(bench[:5]):
        if mon is not None and mon.species in available_switch_species:
            mask[4 + i] = True

    return mask

def action_to_order(action_idx: int, battle):
    """Convert a model action index (0..8) back to a poke-env BattleOrder"""
    active = battle.active_pokemon

    if action_idx < 4:
        sorted_move_ids = sorted((getattr(active, "moves", None) or {}).keys())
        if action_idx >= len(sorted_move_ids):
            return None
        move_id = sorted_move_ids[action_idx]
        move = active.moves[move_id]
        return Player.create_order(move)
    else:
        bench_idx = action_idx - 4
        bench = [mon for key, mon in sorted(battle.team.items()) if mon is not active]
        if bench_idx >= len(bench):
            return None
        target = bench[bench_idx]
        return Player.create_order(target)
    
# ==============================================
# Transition record
# ==============================================

class Transition:
    """Single (s, a, log_prob, V, r, done, mask) record. Lightweight: numpy/CPU"""
    __slots__ = ("obs", "action", "log_prob", "value", "reward", "done", "action_mask")

    def __init__(self, obs, action, log_prob, value, reward, done, action_mask):
        self.obs = obs
        self.action = action
        self.log_prob = log_prob
        self.value = value
        self.reward = reward
        self.done = done
        self.action_mask = action_mask

# ==============================================
# Gen1RLAgent (trainable PPO Player)
# ==============================================

class Gen1Agent(Player):
    """
    poke-env Player wrapping the PokemonAgent network for PPO self-play

    Per turn (choose_move):
        1. embed_battle -> obs dict
        2. forward through PokemonAgent.get_action_and_value
        3. record (obs, action, log_prob, V, mask) into self.trajectory
        4. action_to_order -> return BattleOrder

    Reward:
        Every intermediate turn: 0
        Battle end:              +1 / -1 / 0 written into LAST transition,
                                 done=True set on the same transition
    """
    def __init__(self, model,
                 species_to_idx: Dict[str, int],
                 move_to_idx: Dict[str, int],
                 device: str = "cpu",
                 battle_format="gen1randombattle",
                 deterministic: bool = False,
                 **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self.model = model
        self.species_to_idx = species_to_idx
        self.move_to_idx = move_to_idx
        self.device = torch.device(device)
        self.deterministic = deterministic

        self.trajectory: List[Transition] = []  # store transitions for training
        self._completed_trajectories: List[List[Transition]] = []

    # most important function to change about the user
    def choose_move(self, battle):
        state = embed_battle(battle, self.species_to_idx, self.move_to_idx)
        obs_dev = {k: v.to(self.device) for k, v in state.items()}

        # Forward pass through model
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(
                obs_dev["species_indices"],
                obs_dev["move_indices"],
                obs_dev["numeric_features"],
                obs_dev["context_features"],
                obs_dev["action_mask"],
            )

            if self.deterministic:
                action_idx = int(torch.argmax(logits, dim=-1).item())
                log_prob = float(torch.log_softmax(logits, dim=-1)[0, action_idx].item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action_t = dist.sample()
                action_idx = int(action_t.item())
                log_prob = float(dist.log_prob(action_t).item())
            
            value_f = float(value.item())

        # Store transition
        self.trajectory.append(Transition(
            obs={k: v.cpu() for k, v in state.items()},
            action=action_idx,
            log_prob=log_prob,
            value=value_f,
            reward=0.0,
            done=False,
            action_mask=state["action_mask"].cpu(),
        ))

        # Map action index → legal move
        legal_moves = battle.available_moves
        if not bool(state["action_mask"][0, action_idx]):
            return self.choose_random_move(battle)
        if len(legal_moves) == 0:
            return self.choose_random_move(battle)
        
        order = action_to_order(action_idx, battle)
        if order is None:
            return self.choose_random_move(battle)

        return order # returns BattleOrder (either move or switch to bench mon)
    
    # reward and end-of-battle handling
    def calc_reward(self, battle) -> float:
        """Sparse reward: +1 win / -1 loss / 0 otherwise"""
        if battle.won:
            return 1.0
        if battle.lost:
            return -1.0
        return 0.0
    
    def _battle_finished_callback(self, battle):
        if self.trajectory:
            terminal_reward = self.calc_reward(battle)
            last = self.trajectory[-1]
            last.reward = terminal_reward
            last.done = True
            self._completed_trajectories.append(self.trajectory)
        self.trajectory = []
        try:
            super()._battle_finished_callback(battle)
        except Exception:
            pass

    # trainer-facing API
    def pop_trajectories(self) -> List[List[Transition]]:
        out = self._completed_trajectories
        self._completed_trajectories = []
        return out

class TestingFeatureExtractionAgent(Player):
    def choose_move(self, battle):
        # ---- Global state features ----
        print("=== Global Battle Features ===")

        # Weather
        print("Sun:", int(battle.weather == "sun"))
        print("Rain:", int(battle.weather == "rain"))
        print("Hail:", int(battle.weather == "hail"))
        print("Sandstorm:", int(battle.weather == "sand"))
        print("No weather:", int(battle.weather is None))

        # Stealth Rock / Spikes / Toxic Spikes
        print("Side conditions (your side):", battle.side_conditions)
        print("Side conditions (opponent):", battle.opponent_side_conditions)

        # ---- Active Pokémon features ----
        active_pokemon = battle.active_pokemon
        if active_pokemon:
            print("\n=== Active Pokémon Features ===")
            print("Species:", active_pokemon.species)
            print("Available moves this turn:")
            for move in battle.available_moves:
                print(f" - {move.id}, PP: {move.current_pp}/{move.max_pp}, Target: {move.target}")
            print("All moves in moveset:", active_pokemon.moves)
            print("Current HP fraction:", active_pokemon.current_hp_fraction)
            print("Status:", active_pokemon.status)
            print("Boosts:", active_pokemon.boosts)
            print("Types:", active_pokemon.types)
            print("Gender:", active_pokemon.gender)
            print("Is active:", active_pokemon.active)
            print("Must recharge:", active_pokemon.must_recharge)
            print("Protect count:", active_pokemon.protect_counter)
            print("Last move:", active_pokemon.last_move)

        # ---- Opponent Pokémon features (current active) ----
        opp_active = battle.opponent_active_pokemon
        if opp_active:
            print("\n=== Opponent Active Pokémon Features ===")
            print("Species:", opp_active.species)
            print("Available moves this turn:")
            for move in battle.available_moves:
                print(f" - {move.id}, PP: {move.current_pp}/{move.max_pp}, Target: {move.target}")
            print("All moves in moveset:", opp_active.moves)
            print("Current HP fraction:", opp_active.current_hp_fraction)
            print("Status:", opp_active.status)
            print("Boosts:", opp_active.boosts)
            print("Types:", opp_active.types)
            print("Gender:", opp_active.gender)
            print("Is active:", opp_active.active)
            print("Must recharge:", opp_active.must_recharge)
            print("Protect count:", opp_active.protect_counter)
            print("Last move:", opp_active.last_move)

        # ---- Choose a move ----
        # For now, just pick a random valid move
        return self.choose_random_move(battle)
    
# ==========================================================================
# Layout self-test (run `python env.py`)
# ==========================================================================
 
if __name__ == "__main__":
    print(f"Numeric layout offsets:")
    print(f"  HP            {OFF_HP:4d} +{HP_BINS}")
    print(f"  Status        {OFF_STATUS:4d} +{NUM_STATUS_SLOTS}")
    print(f"  Toxic counter {OFF_TOX_CNT:4d} +{TOXIC_COUNTER_BINS}")
    print(f"  Sleep counter {OFF_SLP_CNT:4d} +{SLEEP_COUNTER_BINS}")
    print(f"  Boosts        {OFF_BOOSTS:4d} +{NUM_BOOST_STATS * BOOST_LEVELS}")
    print(f"  Move PP       {OFF_PP:4d} +{NUM_MOVES_PER_POKEMON * PP_BINS}")
    print(f"  Types         {OFF_TYPES:4d} +{NUM_TYPES}")
    print(f"  Binary flags  {OFF_FLAGS:4d} +6")
    print(f"  Volatile      {OFF_VOLATILE:4d} +8")
    print(f"  TOTAL         {TOTAL_NUMERIC}  (expected {NUM_NUMERIC_FEATURES})")
 
    # Smoke test the encoders with a None mon (unknown slot)
    vec = _pokemon_numeric_features(None, is_active=False, is_opponent=True, is_unknown=True)
    assert vec.shape == (162,)
    assert vec[OFF_FLAGS + FLAG_IS_UNKNOWN] == 1.0
    assert vec[OFF_FLAGS + FLAG_IS_OPPONENT] == 1.0
    assert vec.sum() == 2.0, f"unknown slot should only set 2 flags, got {vec.sum()}"
    print("\n[ok] unknown-slot encoder produces correct sparse vector")
    print("[ok] all assertions passed")