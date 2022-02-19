from functools import total_ordering
import os
from pickletools import optimize 
import torch
import copy
import random
import gym
import gym.spaces 
from collections import deque, namedtuple
import numpy as np 
from torch.utils.tensorboard import SummaryWriter

from atari_wrappers import generate_env
from dqn import DQN_Conv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Experience = namedtuple(
    "Experience", field_names=[
        "state", "action", "reward", "done", "next_state",
    ]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        indices = np.random.choice(
            len(self.buffer), batch_size, replace=False
        )
        states = [self.buffer[idx][0] for idx in indices]
        actions = [self.buffer[idx][1] for idx in indices]
        rewards = [self.buffer[idx][2] for idx in indices]
        dones = [self.buffer[idx][3] for idx in indices]
        next_states = [self.buffer[idx][4] for idx in indices]
        
        return (np.array(states), np.array(actions), 
               np.array(rewards, dtype=np.float32), 
               np.array(dones, dtype=np.uint8), 
               np.array(next_states))


class Agent:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0 


    def step(self, model, epsilon=0.0):
        done_reward = None
        
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else: 
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(DEVICE)
            q_vals = model(state_v)
            _, act = torch.max(q_vals, dim=1)
            action = int(act.item())
    
        next_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        exp = Experience(self.state, action, reward, done, next_state)
        self.replay_buffer.append(exp)
        self.state = next_state
        
        if done:
            done_reward = self.total_reward
            self._reset()
        
        return done_reward

                  
        
def main():    
    # Hyperparameters 
    eps_start = 1.0
    eps_decay = 0.999985
    eps_min = 0.02
    replay_buffer_size = 10000
    init_experiences = 10000
    sync_target_network = 1000
    batch_size = 32
    learning_rate = 1e-4
    reward_bound = 19.0
    gamma = 0.99
    
    # env_name = "PongDeterministic-v4"
    env_name = "PongNoFrameskip-v4"
    env = generate_env(env_name)
    frame = env.reset()
    
    print(type(frame))
    print(frame.shape)
    print(env.observation_space.shape)
    print(env.action_space.n)
    
    Q_network = DQN_Conv(4, env.action_space.n).to(DEVICE)
    Q_tar_network = DQN_Conv(4, env.action_space.n).to(DEVICE)
    
    replay_buffer = ReplayBuffer(replay_buffer_size)
    agent = Agent(env, replay_buffer)
    
    writer = SummaryWriter(comment="-" + env_name)
    
    eps = eps_start
    
    optimizer = torch.optim.Adam(Q_network.parameters(), lr=learning_rate)
    
    total_rewards = []
    frame_idx = 0 
    best_mean_reward = None 
    
    while True: 
        # Sampling Phase
        frame_idx += 1 
        eps = max(eps * eps_decay, eps_min)
        reward = agent.step(Q_network, eps)
        
        if reward is not None: 
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            print("{}: {} games, mean reward {}, (epsilon {})".format(frame_idx, len(total_rewards), mean_reward, eps))

            writer.add_scalar("epsilon", eps, frame_idx)
            writer.add_scalar("mean_reward_last_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(Q_network.state_dict(), env_name + "-best.pth")
                best_mean_reward = mean_reward

                if best_mean_reward is not None:
                    print(f"Best mean reward updated: {best_mean_reward}")
            
            if mean_reward > reward_bound:
                print(f"Solved in {frame_idx} frames.")
                break
                
        if len(replay_buffer) < init_experiences:
            continue 
        
        if frame_idx % sync_target_network == 0:
            Q_tar_network.load_state_dict(Q_network.state_dict())
            print("Target Network updated.")
        
        # Training Phase
        optimizer.zero_grad()
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, dones, next_states = batch 
        
        states_t = torch.tensor(states).to(DEVICE)
        actions_t = torch.tensor(actions).to(DEVICE)
        rewards_t = torch.tensor(rewards).to(DEVICE)
        dones_t = torch.ByteTensor(dones).to(DEVICE)
        next_states_t = torch.tensor(next_states).to(DEVICE)
        
        state_action_values = Q_network(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        next_state_values = Q_tar_network(next_states_t).max(1)[0]
        
        next_state_values[dones_t] = 0.0 
        next_state_values = next_state_values.detach() 
        
        expected_state_action_values = next_state_values * gamma + rewards_t 
        loss_t = torch.nn.MSELoss()(state_action_values, expected_state_action_values)
        
        loss_t.backward()
        optimizer.step() 
        
    writer.close()
    env.close()


if __name__ == "__main__":
    main()