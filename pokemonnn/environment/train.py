# train.py
import asyncio
import torch
import torch.optim as optim
from model import SimpleModel
from env import Gen1Agent
from poke_env.player import RandomPlayer

async def main():
    # 1️⃣ Create model
    model = SimpleModel(input_size=2, output_size=4)

    # 2️⃣ Create agent
    agent = Gen1Agent(model)

    # 3️⃣ Create opponent (online random AI)
    opponent = RandomPlayer(battle_format="gen1randombattle")

    # 4️⃣ Run battles asynchronously
    print("Starting battles...")
    await agent.battle_against(opponent, n_battles=5)
    print("Battles finished!")

    # 5️⃣ Check collected transitions
    print("Transitions collected:", len(agent.memory))
    for t in agent.memory[:5]:
        print(t)

    # 6️⃣ OPTIONAL: basic training step (dummy reward = 0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    for state, action, reward, next_state in agent.memory:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        pred = model(state_tensor)
        target = pred.clone().detach()
        target[action] = reward  # placeholder

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    asyncio.run(main())