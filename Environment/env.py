# env.py
from poke_env.player import Player
import numpy as np
import torch

class Gen1Agent(Player):
    def __init__(self, model, battle_format="gen1randombattle"):
        super().__init__(battle_format=battle_format)
        self.model = model
        self.memory = []  # store transitions for training

    def embed_battle(self, battle):
        """Convert battle state → numeric vector"""
        # Will need to put your embedding function here for this!
        return np.array([
            battle.active_pokemon.current_hp_fraction,
            battle.active_pokemon.current_hp_fraction
        ], dtype=np.float32)

    # most important function to change about the user
    def choose_move(self, battle):
        state = self.embed_battle(battle)
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Forward pass through model
        action_scores = self.model(state_tensor)
        action_idx = int(torch.argmax(action_scores).item())

        # Map action index → legal move
        legal_moves = battle.available_moves
        if len(legal_moves) == 0:
            return self.choose_random_move(battle)

        move = legal_moves[action_idx % len(legal_moves)]

        # Store transition (reward placeholder = 0)
        self.memory.append((state, action_idx, 0, None))
        return move

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