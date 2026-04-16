"""
train.py
========
Main PPO training loop for the Gen 1 Pokémon agent.

Pipeline:
    1. ROLLOUT: have Gen1RLAgent play N battles vs an opponent (self-play
        with a frozen snapshot, or a fixed heuristic for warmup).
    2. COLLECT: drain trajectories into the RolloutBuffer (which computes GAE).
    3. UPDATE: PPOTrainer.update() runs minibatch SGD over the buffer.
    4. VALIDATE: every K updates, run validate() against Random + MaxDamage.
    5. CHECKPOINT: every C updates, save model + optimizer state.

The opponent for self-play is a *frozen* snapshot  of the policy from a few
updates ago, refreshed every snapshot_interval updates. This is the standard
self-play stabilization tricl: training against your latest self alone
causes oscillation.

Run:
    python train.py
"""

from __future__ import annotations
from poke_env import LocalhostServerConfiguration, ServerConfiguration, SimpleHeuristicsPlayer

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

# ==================================
# Hyperparameters
# ==================================

GAMMA = 0.99
GAE_LAMBDA = 0.95
TOTAL_UPDATES = 1000    # outer PPO iterations
BATTLES_PER_UPDATE = 64 # rollout battles between updates
SNAPSHOT_INTERVAL = 25  # refresh self-play opponent every N updates
VALIDATE_EVERY = 25     # run validation every N updates
CHECKPOINT_EVERY = 50   # save checkpoint every N updates
N_VAL_BATTLES = 50      # battles per opponent during validation
N_INSTANCES = 16        # Number of showdown instances to run parallelized
BATTLES_PER_INSTANCE = BATTLES_PER_UPDATE // N_INSTANCES
RESUME_FROM: Path | None = Path("pokemonnn/training/checkpoints/agent_update_0950.pt") # e.b.g Path("pokemonnn/training/checkpoints/agent_update_0100.pt")

CHECKPOINT_DIR = Path("pokemonnn/training/checkpoints")
EMBEDDINGS_DIR = Path("pokemonnn/network/embeddings")

# ==================================
# Setup helpers
# ==================================

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

# def make_snapshot_opponent(model: PokemonAgent, species_to_idx, move_to_idx,
#                            device: str, battle_format: str) -> Gen1RLAgent:
#     """Frozen deep copy of the current policy used as a self-play opponent"""
#     snap = copy.deepcopy(model)
#     for p in snap.parameters():
#         p.requires_grad_(False)
#     snap.eval()
#     return Gen1RLAgent(
#         model=snap,
#         species_to_idx=species_to_idx,
#         move_to_idx=move_to_idx,
#         device=device,
#         battle_format=battle_format,
#         deterministic=False # opponent should still explore
#     )

def refresh_snapshot_weights(snap_model: PokemonAgent, live_model: PokemonAgent):
    """Copy live model weights into the existing snapshot. No new agents"""
    with torch.no_grad():
        for snap_p, live_p in zip(snap_model.parameters(), live_model.parameters()):
            snap_p.copy_(live_p)
        for snap_b, live_b in zip(snap_model.buffers(), live_model.buffers()):
            snap_b.copy_(live_b)

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

def make_server_config(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

def make_parallel_agents(model, species_to_idx, move_to_idx, device, battle_format,
                         num_instances: int, base_port: int = 8000):
    """
    Create N learner agents and N opponent agents, one pair per Showdown port.
    All learners share the same underlying model (same weights, same gradients).
    Opponents share a frozen snapshot.
    """
    
    # One frozen snapshot shared across all opponent connections.
    snap = copy.deepcopy(model)
    for p in snap.parameters():
        p.requires_grad_(False)
    snap.eval()
    
    learners = []
    opponents = []
    for i in range(num_instances):
        port = base_port + i
        server_cfg = ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )
        
        learner = Gen1RLAgent(
            model=model,
            species_to_idx=species_to_idx,
            move_to_idx=move_to_idx,
            device=device,
            battle_format=battle_format,
            deterministic=False,
            server_configuration=server_cfg,
            # username=f"Learner{i}",
        )
        
        opponent = Gen1RLAgent(
            model=snap,
            species_to_idx=species_to_idx,
            move_to_idx=move_to_idx,
            device=device,
            battle_format=battle_format,
            deterministic=False,
            server_configuration=server_cfg,
            # username=f"Opponent{i}",
        )
        
        learners.append(learner)
        opponents.append(opponent)
    
    return learners, opponents
    

# =====================================
# Training Loop
# =====================================

async def train():
    last_update_time = time.time()
    last_update_steps = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Device: {device}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    species_to_idx, move_to_idx = load_embedding_index()
    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] Trainable parameters: {n_params:,}")

    ppo_cfg = PPOConfig()
    trainer = PPOTrainer(model, ppo_cfg, total_updates=TOTAL_UPDATES, device=device)

    battle_format = "gen1randombattle"

    # Learner agents: trains, samples stochastically
    learners, opponents = make_parallel_agents(model=model,
                                               species_to_idx=species_to_idx,
                                               move_to_idx=move_to_idx,
                                               device=device,
                                               battle_format=battle_format,
                                               num_instances=N_INSTANCES,
                                            )
    
    # learner_agent = Gen1RLAgent(
    #     model=model,
    #     species_to_idx=species_to_idx,
    #     move_to_idx=move_to_idx,
    #     device=str(device),
    #     battle_format=battle_format,
    #     deterministic=False
    # )

    # # Initial self-play opponent = current policy snapshot
    # opponent_agent = make_snapshot_opponent(
    #     model, species_to_idx, move_to_idx, str(device), battle_format
    # )

    global_steps = 0
    start_update = 0

    if RESUME_FROM is not None:
        if RESUME_FROM.exists():
            start_update, global_steps = load_checkpoint(RESUME_FROM, model, trainer)
        else:
            print(f"[resume] WARNING: {RESUME_FROM} not found, starting from scratch.")

    start_time = time.time()

    for update in range(start_update + 1, TOTAL_UPDATES + 1):
        
        t0 = time.time()
        # 1. Rollout across all parallel instances
        for learner in learners: 
            learner.pop_trajectories() # ensure clean slate
            
        await asyncio.gather(*[
            learner.battle_against(opponent, n_battles=BATTLES_PER_INSTANCE)
            for learner, opponent in zip(learners, opponents)
        ])
        
        # Drain opponent's trajectories too (we don't train on them)
        trajectories = []
        for learner, opponent in zip(learners, opponents):
            trajectories.extend(learner.pop_trajectories())     
            opponent.pop_trajectories()
        t1 = time.time()

        # 2. Collect into buffer
        buffer = RolloutBuffer(gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                               normalize_advantages=True)
        for traj in trajectories:
            buffer.add_trajectory(traj)
        batch = buffer.build_batch()
        global_steps += len(batch)
        t2 = time.time()

        # 3. PPO Update
        metrics = trainer.update(batch)
        t3 = time.time()

        # 4. Logging
        elapsed = time.time() - start_time
        sps = global_steps / max(1.0, elapsed)
        
        print(
            f"[update {update:4d}/{TOTAL_UPDATES}] "
            f"steps={global_steps:>7} ({sps:5.1f}/s) "
            f"[t: rollout={t1-t0:.2f} buf={t2-t1:.2f} update={t3-t2:.2f}] "
            f"lr={metrics.get('lr', 0):.2e} "
            f"pg={metrics.get('pg_loss', 0):+.4f} "
            f"v={metrics.get('value_loss', 0):.4f} "
            f"ent={metrics.get('entropy', 0):.3f} "
            f"kl={metrics.get('approx_kl', 0):.4f} "
            f"clipfrac={metrics.get('clip_frac', 0):.3f} "
            f"expvar={metrics.get('explained_variance', 0):+.3f}"
        )

        # 5. Validation
        if update % VALIDATE_EVERY == 0:
            val = await validate(
                model, species_to_idx, move_to_idx,
                n_battles=N_VAL_BATTLES, device=str(device),
                battle_format=battle_format
            )
            print(
                f"[val {update:4d}] "
                f"vs_random={val.vs_random_winrate:.3f} "
                f"vs_maxdmg={val.vs_maxdamage_winrate:.3f} "
                f"vs_simpleheuristic={val.vs_simpleheuristics_winrate:.3f} "
                f"(n={val.n_battles_each})"
            )
        
        # 6. Snapshot opponent refresh
        if update % SNAPSHOT_INTERVAL == 0:
            refresh_snapshot_weights(snap_model=opponents[0].model, live_model=model)
            print(f"[snap {update:4d}] refreshed self-play opponent")

        # 7. Checkpoint
        if update % CHECKPOINT_EVERY == 0:
            ckpt_path = CHECKPOINT_DIR / f"agent_update_{update:04d}.pt"
            torch.save({
                "update": update,
                "global_steps": global_steps,
                "model_state": model.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
            }, ckpt_path)
            print(f"[ckpt {update:4d}] saved -> {ckpt_path}")
        
    print(f"\n[done] training finished. total steps={global_steps:,}. "
            f"elapsed={time.time() - start_time:.1f}s")
        
if __name__ == "__main__":
    asyncio.run(train())