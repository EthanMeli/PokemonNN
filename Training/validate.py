"""
validate.py
===========
Periodic validation: play N battles against fixed reference opponents
(RandomPlayer and a MaxDamagePlayer heuristic) using the current network
in deterministic (argmax) mode, and report win rates.

This is run between PPO updates so we can detect regressions and decide
when to checkpoint. Win rate vs. RandomPlayer should climb fast (> 90%
within a few hundred battles); win rate vs. MaxDamage is the stronger
signal of real learning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict

from poke_env.player import Player, RandomPlayer

from env import Gen1RLAgent

class MaxDamagePlayer(Player):
    """Heuristic baseline: always pick the highest base-power available move."""
    def choose_move(self, battle):
        if battle.available_moves:
            best = max(battle.available_moves, key=lambda m: m.base_power or 0)
            return self.create_order(best)
        return self.choose_random_move(battle)
    
@dataclass
class ValidationResult:
    vs_random_winrate: float
    vs_maxdamage_winrate: float
    n_battles_each: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "val/vs_random_winrate": self.vs_random_winrate,
            "val/vs_maxdamage_winrate": self.vs_maxdamage_winrate,
            "val/n_battles": self.n_battles_each,
        }
    
async def _play_n(agent: Gen1RLAgent, opponent: Player, n_battles: int) -> float:
    start_wins = agent.n_won_battles
    start_total = agent.n_finished_battles
    await agent.battle_against(opponent, n_battles=n_battles)
    delta_wins = agent.n_won_battles - start_wins
    delta_total = agent.n_finished_battles - start_total
    return float(delta_wins) / max(1, delta_total)

async def validate(model, species_to_idx, move_to_idx,
                   n_battles: int = 20, device: str = "cpu",
                   battle_format: str = "gen1randombattle") -> ValidationResult:
    """
    Spin up a deterministic validation agent and run it against two opponents.
    Trajectories collected during validation are dropped (not used for training).
    """
    val_agent = Gen1RLAgent(
        model=model,
        species_to_idx=species_to_idx,
        move_to_idx=move_to_idx,
        device=device,
        battle_format=battle_format,
        deterministic=True, # argmax for stable evaluation
    )

    random_opp = RandomPlayer(battle_format=battle_format)
    maxdmg_opp = MaxDamagePlayer(battle_format=battle_format)

    vs_random = await _play_n(val_agent, random_opp, n_battles)
    vs_maxdmg = await _play_n(val_agent, maxdmg_opp, n_battles)

    # Drop trajectories collected during validation
    val_agent.pop_trajectories()

    return ValidationResult(
        vs_random_winrate=vs_random,
        vs_maxdamage_winrate=vs_maxdmg,
        n_battles_each=n_battles,
    )