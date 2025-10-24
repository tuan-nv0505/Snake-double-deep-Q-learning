from idlelib.pyparse import trans
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))


from utils.settings import GRID_SIZE
import torch
from torch import nn
from torchvision.transforms import ToTensor
from environment.environment import Environment
from collections import deque
from agent.deep_q_network import DeepQNetwork
import numpy as np
import random

EPISODES = 5000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
GAMMA = 0.99
LR = 0.0005
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 20

def select_action(state, policy_net, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(3)
    else:
        with torch.no_grad():
            q_values = policy_net(state.unsqueeze(0))
            return torch.argmax(q_values).item()

def main():
    env = Environment(GRID_SIZE)
    policy_net = DeepQNetwork(3)
    target_net = DeepQNetwork(3)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.MSELoss()

    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START

    transforms = ToTensor()
    for episode in range(EPISODES):
        state = transforms(env.reset())
        done = False
        total_reward = 0
        total_loss = []
        score = 0

        while not done:
            action = select_action(state, policy_net, epsilon)
            next_state, reward, done = env.step(action)
            next_state = transforms(next_state)
            total_reward += reward

            memory.append((state, action, reward, next_state, done))
            state = next_state

            # Train
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards)
                next_states = torch.stack(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Q(s, a)
                q_values = policy_net(states).gather(1, actions)
                next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1).detach()

                # Q_target = r + γ * Q_target_net(s', a')
                targets = rewards + (1 - dones) * GAMMA * next_q_values

                loss = criterion(q_values.squeeze(), targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    total_loss.append(loss.item())

        # Epsilon decay
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if total_loss:
            print(f"Episode [{episode + 1}][{EPISODES}] | Loss: {sum(total_loss)/len(total_loss)} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

        if (episode + 1) % 50 == 0 and episode != 0:
            torch.save(policy_net.state_dict(), "agent/double_dqn_snake.pth")
            print(f"Episode: {episode + 1} Model saved.")

if __name__ == '__main__':
    main()