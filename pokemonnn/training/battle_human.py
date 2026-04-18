"""
battle_human.py
===============
Script used to challenge a specific user.
"""

from __future__ import annotations
from poke_env import AccountConfiguration, ShowdownServerConfiguration

import asyncio
import copy
import os
import time
from pathlib import Path

import torch

from pokemonnn.network.pokemon_encoder import Gen1Config
from pokemonnn.network.battle_model import PokemonAgent, TransformerConfig
from pokemonnn.environment.env import Gen1RLAgent
from .rollout_buffer import RolloutBuffer
from .ppo_trainer import PPOConfig, PPOTrainer
from .validate import validate

RESUME_FROM: Path | None = Path("pokemonnn/training/checkpoints/agent_update_0060.pt") # e.b.g Path("pokemonnn/training/checkpoints/agent_update_0100.pt")
CHECKPOINT_DIR = Path("pokemonnn/training/checkpoints")
EMBEDDINGS_DIR = Path("pokemonnn/network/embeddings")
START_UPDATE = 60
UPDATES = 100

def build_model(device: torch.device) -> PokemonAgent:
    gen1_cfg = Gen1Config()
    tf_cfg = TransformerConfig()

    species_matrix = move_matrix = unknown_emb = None
    if EMBEDDINGS_DIR.exists():
        sp_path = EMBEDDINGS_DIR / "species_matrix.pt"
        mv_path = EMBEDDINGS_DIR / "move_matrix.pt"
        un_path = EMBEDDINGS_DIR / "unknown_embedding.pt"
        if sp_path.exists() and mv_path.exists() and un_path.exists():
            species_matrix = torch.load(sp_path, weights_only=True, map_location="cpu")
            move_matrix = torch.load(mv_path, weights_only=True, map_location="cpu")
            unknown_emb = torch.load(un_path, weights_only=True, map_location="cpu")
            print("[setup] Loaded pretrained LLM embeddings.")
        else:
            print("[setup] Embedding files missing - using random init for LLM embeddings.")
    else:
        print("[setup] No embeddings dir - using random init for LLM embeddings.")

    model = PokemonAgent(gen1_cfg, tf_cfg, species_matrix, move_matrix, unknown_emb)
    model.to(device)
    return model

def load_embedding_index():
    idx_path = EMBEDDINGS_DIR / "embedding_index.pt"
    if idx_path.exists():
        idx = torch.load(idx_path, weights_only=True)
        return idx["species_to_idx"], idx["move_to_idx"]
    print("[setup] No embedding_index.pt - using empty mappings (all species/moves unknown).")
    return {}, {}

def load_checkpoint(ckpt_path: Path, model: PokemonAgent,
                    trainer: PPOTrainer) -> tuple[int, int]:
    """
    Restore model weights, optimizer state, update counter, and global step
    counter from a saved checkpoint. Returns (start_update, global_steps)

    Also rewinds the LR annealing schedule to the correct point by setting
    trainer._update_idx so LR resumes where it left off.
    """
    print(f"[resume] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=trainer.device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state"])

    start_update = int(ckpt.get("update", 0))
    global_steps = int(ckpt.get("global_steps", 0))

    trainer._update_idx = start_update
    print(f"[resume] Restored: update={start_update}, "
          f"global_steps={global_steps:,}")
    return start_update, global_steps

async def main():    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    species_to_idx, move_to_idx = load_embedding_index()
    model = build_model(device)
    
    ppo_cfg = PPOConfig(
        learning_rate=3e-4,
        anneal_lr=False,
    )
    trainer = PPOTrainer(model, ppo_cfg, total_updates=UPDATES, device=device)
    
    player1 = Gen1RLAgent(
        model=model,
        species_to_idx=species_to_idx,
        move_to_idx=move_to_idx,
        device=device,
        battle_format="gen1randombattle",
        deterministic=False,
        server_configuration=ShowdownServerConfiguration,
        account_configuration=AccountConfiguration("edogmeli", "Edoge1234"),
        start_timer_on_battle_start=True,
    )
    
    player2 = Gen1RLAgent(
        model=model,
        species_to_idx=species_to_idx,
        move_to_idx=move_to_idx,
        device=device,
        battle_format="gen1randombattle",
        deterministic=False,
        server_configuration=ShowdownServerConfiguration,
        account_configuration=AccountConfiguration("lxrelite88", "Edoge1234"),
        start_timer_on_battle_start=True,
    )
    
    player3 = Gen1RLAgent(
        model=model,
        species_to_idx=species_to_idx,
        move_to_idx=move_to_idx,
        device=device,
        battle_format="gen1randombattle",
        deterministic=False,
        server_configuration=ShowdownServerConfiguration,
        account_configuration=AccountConfiguration("pokenn882", "Edoge1234"),
        start_timer_on_battle_start=True,
    )
    
    player4 = Gen1RLAgent(
        model=model,
        species_to_idx=species_to_idx,
        move_to_idx=move_to_idx,
        device=device,
        battle_format="gen1randombattle",
        deterministic=False,
        server_configuration=ShowdownServerConfiguration,
        account_configuration=AccountConfiguration("pokenn888", "Edoge1234"),
        start_timer_on_battle_start=True,
    )
    
    player5 = Gen1RLAgent(
        model=model,
        species_to_idx=species_to_idx,
        move_to_idx=move_to_idx,
        device=device,
        battle_format="gen1randombattle",
        deterministic=False,
        server_configuration=ShowdownServerConfiguration,
        account_configuration=AccountConfiguration("pokenn883", "Edoge1234"),
        start_timer_on_battle_start=True,
    )
    
    player6 = Gen1RLAgent(
        model=model,
        species_to_idx=species_to_idx,
        move_to_idx=move_to_idx,
        device=device,
        battle_format="gen1randombattle",
        deterministic=False,
        server_configuration=ShowdownServerConfiguration,
        account_configuration=AccountConfiguration("pokenn884", "Edoge1234"),
        start_timer_on_battle_start=True,
    )
    
    player7 = Gen1RLAgent(
        model=model,
        species_to_idx=species_to_idx,
        move_to_idx=move_to_idx,
        device=device,
        battle_format="gen1randombattle",
        deterministic=False,
        server_configuration=ShowdownServerConfiguration,
        account_configuration=AccountConfiguration("pokenn885", "Edoge1234"),
        start_timer_on_battle_start=True,
    )
    
    player8 = Gen1RLAgent(
        model=model,
        species_to_idx=species_to_idx,
        move_to_idx=move_to_idx,
        device=device,
        battle_format="gen1randombattle",
        deterministic=False,
        server_configuration=ShowdownServerConfiguration,
        account_configuration=AccountConfiguration("pokenn886", "Edoge1234"),
        start_timer_on_battle_start=True,
    )
    
    players = [player1, player2, player3, player4, player5, player6, player7, player8]
    
    if RESUME_FROM is not None:
        if RESUME_FROM.exists():
            start_update, global_steps = load_checkpoint(RESUME_FROM, model, trainer)
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = ppo_cfg.learning_rate
            trainer._update_idx = 0
            print(f"[resume] LR forced to {ppo_cfg.learning_rate} for fine-tuning.")
        else:
            print(f"[resume] WARNING: {RESUME_FROM} not found, starting from scratch.")
            global_steps = 0
            
    for update in range(START_UPDATE+1, UPDATES+1):
        for player in players:
            player.pop_trajectories()
        
        await asyncio.gather(*[
            player.ladder(2)
            for player in players
        ])  
        
        trajectories = []
        for player in players:
            trajectories.extend(player.pop_trajectories())
        
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95, normalize_advantages=True)
        for traj in trajectories:
            buffer.add_trajectory(traj)
        batch = buffer.build_batch()
        
        global_steps += sum(len(t) for t in trajectories)
        
        metrics = trainer.update(batch)
        
        print(
            f"[update {update:4d}/{UPDATES}] "
            f"steps = {global_steps} "
            f"lr={metrics.get('lr', 0):.2e} "
            f"pg={metrics.get('pg_loss', 0):+.4f} "
            f"v={metrics.get('value_loss', 0):.4f} "
            f"ent={metrics.get('entropy', 0):.3f} "
            f"kl={metrics.get('approx_kl', 0):.4f} "
            f"clipfrac={metrics.get('clip_frac', 0):.3f} "
            f"expvar={metrics.get('explained_variance', 0):+.3f}"
        )
        
        if update % 1 == 0:
            val = await validate(
                model, species_to_idx, move_to_idx,
                n_battles=50, device=str(device),
                battle_format="gen1randombattle"
            )
            print(
                f"[val {update:4d}] "
                f"vs_random={val.vs_random_winrate:.3f} "
                f"vs_maxdmg={val.vs_maxdamage_winrate:.3f} "
                f"vs_simpleheuristic={val.vs_simpleheuristics_winrate:.3f} "
                f"(n={val.n_battles_each})"
            )
            
        if update % 2 == 0:
            ckpt_path = CHECKPOINT_DIR / f"agent_update_{update:04d}.pt"
            torch.save({
                "update": update,
                "global_steps": global_steps,
                "model_state": model.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
            }, ckpt_path)
            print(f"[ckpt {update:4d}] saved -> {ckpt_path}")
        
    i = 1
    for player in players:
        print(f"Player {i}: ")
        for battle in player.battles.values():
            print(battle.rating, battle.opponent_rating)
        i += 1
    print(f"\n[done] training finished. total steps={global_steps:,}. ")
    
if __name__ == "__main__":
    asyncio.run(main())