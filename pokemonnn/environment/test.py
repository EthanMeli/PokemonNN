import asyncio

from model import SimpleModel

from env import Gen1Agent, TestingFeatureExtractionAgent
from poke_env.player import RandomPlayer
from poke_env.player import Player
class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # Chooses a move with the highest base power when possible
        if battle.available_moves:
            # Iterating over available moves to find the one with the highest base power
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            # Creating an order for the selected move
            return self.create_order(best_move)
        else:
            # If no attacking move is available, perform a random switch
            # This involves choosing a random move, which could be a switch or another available action
            return self.choose_random_move(battle)


async def main():
    # agent = Gen1Agent(SimpleModel(input_size=2, output_size=4))
    agent = TestingFeatureExtractionAgent(battle_format="gen1randombattle")
    opponent_1 = RandomPlayer(battle_format="gen1randombattle")
    opponent_2 = MaxDamagePlayer(battle_format="gen1randombattle")

    await agent.battle_against(opponent_1, n_battles=1)

    print(f"Finished battles: {agent.n_finished_battles}")
    print(f"Player 1 wins: {agent.n_won_battles}")


if __name__ == "__main__":
    asyncio.run(main())
