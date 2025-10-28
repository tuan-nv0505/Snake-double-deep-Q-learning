from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import torch
from torch import nn
from environment.environment import Environment
from collections import deque
from agent.deep_q_network import DeepQNetwork
import numpy as np
from utils.utils import get_args
import random
import os
import shutil
from torch.utils.tensorboard import SummaryWriter


def select_action(state, agent, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.randint(3)
        return action
    else:
        with torch.no_grad():
            q_values = agent(state.unsqueeze(0))
            action = torch.argmax(q_values).item()
            return action

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment(args.grid_size, args.epsilon_start)

    if os.path.exists('checkpoint') and args.reset_checkpoint:
        shutil.rmtree('checkpoint')

    policy_net = DeepQNetwork(3).to(device)
    if os.path.exists('checkpoint/snake_dqn.pth'):
        policy_net.load_state_dict(torch.load('checkpoint/snake_dqn.pth'))
        print('Load file snake_dqn.pth successfully. Continue training.')
    else:
        print('Load file snake_dqn.pth failed.')
    target_net = DeepQNetwork(3).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

    memory_replay = deque(maxlen=args.memory_replay)
    epsilon = args.epsilon_start
    loss_min = sys.float_info.max

    if os.path.exists(args.logging) and args.reset_logging:
        shutil.rmtree(args.logging)
    writer = SummaryWriter(args.logging)

    num_batches = 0
    for episode in range(args.episodes):
        frame = env.reset()
        stack_frames = deque([torch.from_numpy(frame).to(device)] * args.frame_size, maxlen=args.frame_size)
        memory_episode = []

        while not env.done:
            state = torch.stack(list(stack_frames))

            action = select_action(state, policy_net, epsilon)
            next_frame, reward, done, score = env.step(action)
            stack_frames.append(torch.from_numpy(next_frame).to(device))

            next_state = torch.stack(list(stack_frames))

            memory_replay.append((state, action, reward, done, next_state))
            memory_episode.append((state, action, score, reward, done, next_state))

            if len(memory_replay) >= args.batch:
                num_batches += 1
                batch = random.sample(memory_replay, args.batch)
                state_batch, action_batch, reward_batch, done_batch, next_state_batch = zip(*batch)

                state_batch = torch.stack(state_batch)
                action_batch = torch.tensor(action_batch).unsqueeze(1).to(device)
                reward_batch = torch.tensor(reward_batch).unsqueeze(1).to(device)
                done_batch = torch.tensor(tuple(map(int, done_batch))).unsqueeze(1).to(device)
                next_state_batch = torch.stack(next_state_batch)

                q_values = policy_net(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    next_action_batch = torch.argmax(policy_net(next_state_batch), dim=1, keepdim=True)
                    next_q_values = target_net(next_state_batch).gather(1, next_action_batch)
                    targets = reward_batch + (1 - done_batch) * args.gamma * next_q_values

                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for t_param, p_param in zip(target_net.parameters(), policy_net.parameters()):
                        t_param.data.copy_(t_param.data * args.tau + p_param.data * (1.0 - args.tau))

                writer.add_scalar('Replay Memory/Loss', loss.item(), num_batches)

        state, action, score, reward, done, next_state = zip(*memory_episode)
        total_reward = sum(reward)
        total_score = sum(score)
        state = torch.stack(state)
        action = torch.tensor(action).unsqueeze(1).to(device)
        reward = torch.tensor(reward).unsqueeze(1).to(device)
        done = torch.tensor(tuple(map(int, done))).unsqueeze(1).to(device)
        next_state = torch.stack(next_state)

        with torch.no_grad():
            q_values = policy_net(state).gather(1, action)
            next_action = torch.argmax(policy_net(next_state), dim=1, keepdim=True)
            next_q_values = target_net(next_state).gather(1, next_action)
            targets = reward + (1 - done) * args.gamma * next_q_values
            loss = criterion(q_values, targets)

        writer.add_scalar('Episode/Loss', loss.item(), episode)
        writer.add_scalar('Episode/Reward', total_reward, episode)
        writer.add_scalar('Episode/Score', total_score, episode)
        writer.add_scalar('Epsilon', epsilon, episode)

        print(f"[{episode + 1}][{args.episodes}] | Epsilon: {epsilon} | Loss: {loss.item():.2f} | Reward: {total_reward} | Score: {total_score}")
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        env.epsilon = epsilon

        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')

        torch.save(policy_net.state_dict(), 'checkpoint/snake_dqn.pth')
        if loss.item() < loss_min:
            print('Update best checkpoint.')
            loss_min = loss.item()
            torch.save(policy_net.state_dict(), 'checkpoint/best_snake_dql.pth')


if __name__ == '__main__':
    args = get_args()
    main(args)