from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from environment.environment import Environment
from agent.deep_q_network import DeepQNetwork
from utils.utils import get_args

import torch
import numpy as np
from collections import deque
import random
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import shutil

torch.set_printoptions(threshold=torch.inf)
def select_action(state, agent, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.randint(3)
        return action
    else:
        with torch.no_grad():
            q_values = agent(state.unsqueeze(0))
            action = torch.argmax(q_values).item()
            return action

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment(args.grid_size)
    policy_net = DeepQNetwork().to(device)
    target_net = DeepQNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    memory_replay = deque(maxlen=50000)
    epsilon = args.epsilon_start
    loss_min = sys.float_info.max
    writer = SummaryWriter(args.logging)

    num_batches = 0
    for episode in range(args.episodes):
        env.reset()
        total_reward = 0
        total_score = 0
        memory_episode = []

        while not env.done:
            state = torch.from_numpy(env.get_state()).to(device)
            action = select_action(state, policy_net, epsilon)
            next_state, reward, done, score = env.step(action, epsilon)
            next_state = torch.from_numpy(next_state).to(device)
            memory_replay.append((state, action, reward, done, next_state))
            memory_episode.append((state, action, reward, done, next_state))

            total_reward += reward
            total_score += score

            if len(memory_replay) >= args.batch:
                sample = random.sample(memory_replay, args.batch)
                state_batch, action_batch, reward_batch, done_batch, next_state_batch = zip(*sample)
                state_batch = torch.stack(state_batch)
                action_batch = torch.tensor(action_batch).to(device).unsqueeze(1)
                reward_batch = torch.tensor(reward_batch).to(device).unsqueeze(1)
                done_batch = torch.tensor(tuple(map(int, done_batch))).to(device).unsqueeze(1)
                next_state_batch = torch.stack(next_state_batch)

                q_values = policy_net(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    next_action = torch.argmax(policy_net(next_state_batch), dim=1, keepdim=True)
                    next_q_values = target_net(next_state_batch).gather(1, next_action)
                    q_target = reward_batch + (1 - done_batch) * args.gamma * next_q_values

                    for t_param, p_param in zip(target_net.parameters(), policy_net.parameters()):
                        t_param.data.copy_((1 - args.tau) * t_param + args.tau * p_param)

                loss = criterion(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Loss/Replay Buffer', loss.item(), num_batches)
                num_batches += 1

        state_episode, action_episode, reward_episode, done_episode, next_state_episode = zip(*memory_episode)
        state_episode = torch.stack(state_episode)
        action_episode = torch.tensor(action_episode).to(device).unsqueeze(1)
        reward_episode = torch.tensor(reward_episode).to(device).unsqueeze(1)
        done_episode = torch.tensor(tuple(map(int, done_episode))).to(device).unsqueeze(1)
        next_state_episode = torch.stack(next_state_episode)

        with torch.no_grad():
            q_values = policy_net(state_episode).gather(1, action_episode)
            next_action = torch.argmax(policy_net(next_state_episode), dim=1, keepdim=True)
            next_q_values = target_net(next_state_episode).gather(1, next_action)
            q_target = reward_episode + (1 - done_episode) * args.gamma * next_q_values

            loss = criterion(q_values, q_target)

        print(f"[{episode + 1}][{args.episodes}] | Epsilon: {epsilon} | Loss: {loss.item():.2f} | Reward: {total_reward} | Score: {total_score}")
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        env.epsilon = epsilon

        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')

        if episode % 10 == 0 and episode != 0:
            print('Save snake_dqn.pth.')
            torch.save(policy_net.state_dict(), 'checkpoint/snake_dqn.pth')
        if loss.item() < loss_min:
            print('Update best checkpoint.')
            loss_min = loss.item()
            torch.save(policy_net.state_dict(), 'checkpoint/best_snake_dqn.pth')

        writer.add_scalar('Episode/Reward', total_reward, episode)
        writer.add_scalar('Episode/Score', total_score, episode)
        writer.add_scalar('Episode/Epsilon', epsilon, episode)
        writer.add_scalar('Loss/Episode', loss.item(), episode)

if __name__ == '__main__':
    if os.path.exists('tensorboard'):
        shutil.rmtree('tensorboard')
    args = get_args()
    train(args)