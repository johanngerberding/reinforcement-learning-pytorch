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
    def __init__(
        self, 
        actions, 
        lr: float, 
        eps_start: float,
        eps_decay: float,
        eps_min: float, 
        num_frames: int = 4, 
        gamma: float = 0.95, 
        max_len_memory: int = 10000
    ):
        self.memory = ReplayBuffer(max_len_memory)
        self.actions = actions 
        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma 
        self.lr = lr 
        self.num_frames = num_frames
        self.total_timesteps = 0
        self.Q = DQN_Conv(self.num_frames, len(actions)).to(DEVICE)
        self.Q_tar = DQN_Conv(self.num_frames, len(actions)).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
    
    def sample_action(self, state):
        "Epsilon-greedy action selection."
        # Explore
        if np.random.rand() < self.eps:
            return random.sample(self.actions, 1)[0]

        # Greedy action
        act_idx = torch.argmax(self.Q(state)).cpu().numpy()
        return self.actions[act_idx]

    def train(self, batch):
        states, actions, rewards, dones, next_states = batch 
        states = torch.tensor(states).to(DEVICE)
        next_states = torch.tensor(next_states).to(DEVICE)
        actions = torch.tensor(actions).to(DEVICE)
        rewards = torch.tensor(rewards).to(DEVICE)
        done_mask = torch.ByteTensor(dones).to(DEVICE)
        
        state_action_values = self.Q(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        print(state_action_values.size())
        next_state_values = self.Q_tar(next_states).max(1)[0]
        print(next_state_values.size())
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        # calculate y with bellman equation
        expected_state_action_values = next_state_values * self.gamma + rewards
        
        loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def sync(self):
        "Sync target network"
        self.Q_tar.load_state_dict(self.Q.state_dict())
        
        
def main():    
    # Hyperparameters 
    eps_start=1.0
    eps_decay=.999985
    eps_min=0.02
    init_experiences = 10000
    sync_target_network = 10000
    batch_size = 32
    
    env_name = "PongDeterministic-v4"
    
    env = generate_env(env_name)
    frame = env.reset()
    
    print(type(frame))
    print(frame.shape)
    
    print(env.action_space.n)
    print(env.unwrapped.get_action_meanings())
    print(env.action_space)
    print(env.observation_space.shape)
    
    # SAMPLING PHASE
    # fill up the Replay Buffer by interaction
    actions = [x for x in range(env.action_space.n)]
    agent = Agent(actions, 0.003, eps_start, eps_decay, eps_min)
    state = env.reset()
    
            
    while True:
        # sample phase
        state = torch.tensor(state).to(DEVICE)
        action = agent.sample_action(state)
        next_state, reward, done, info = env.step(action)
        state = state.cpu().numpy()
        exp = Experience(state, action, reward, done, next_state)
        agent.memory.append(exp)
        state = next_state
        frame_idx = 0
        
        if len(agent.memory) >= init_experiences:
            print(f"Saved {init_experiences} experiences.")
            
            frame_idx += 1
            agent.eps = max(agent.eps * agent.eps_decay, agent.eps_min)
            
            # learning phase 
            batch = agent.memory.sample(batch_size)
            loss = agent.train(batch)
            
            if frame_idx % sync_target_network == 0:
                agent.sync()
            break
    
    env.close()


if __name__ == "__main__":
    main()