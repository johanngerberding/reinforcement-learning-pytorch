import os 
import torch
import copy
import random
import gym
import gym.spaces 
from collections import deque
import numpy as np 
import cv2

from dqn import DQN_Conv


class Agent:
    def __init__(
        self, 
        actions, 
        lr: float, 
        epsilon_start: float, 
        num_frames: int = 4, 
        gamma: float = 0.95, 
        max_len_memory: int = 1000000
    ):
        self.memory = Memory(max_len=max_len_memory)
        self.actions = actions 
        self.epsilon = epsilon_start
        self.gamma = gamma 
        self.lr = lr 
        self.num_frames = num_frames
        self.total_timesteps = 0
        self.Q = DQN_Conv(self.num_frames, len(actions))
        self.Q_tar = copy.deepcopy(self.Q)
    
    def sample_action(self, state):
        "Epsilon-greedy action selection."
        # Explore
        if np.random.rand() < self.epsilon:
            return random.sample(self.actions, 1)[0]

        # Greedy action
        act_idx = torch.argmax(self.Q(state)).cpu().numpy()
        return self.actions[act_idx]
        


class Memory:
    def __init__(self, max_len: int):
        self.max_len = max_len 
        self.frames = deque(maxlen=self.max_len)
        self.actions = deque(maxlen=self.max_len)
        self.rewards = deque(maxlen=self.max_len)
        self.dones = deque(maxlen=self.max_len)
    
    def add_exp(self, frame, reward, action, done):
        self.frames.append(frame)
        self.rewards.append(reward)
        self.actions.append(action)
        self.dones.append(done)
        

def preprocess_frame(frame: np.array, size: tuple = (84, 84)) -> np.array:
    "Preprocess each frame: Crop, grayscale, resize"
    frame = frame[30:-12,5:-4]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
    return frame


def init_game(name, env, agent):
    env.reset()
    frame = preprocess_frame(env.step(0)[0])
    dummy_action = 0
    dummy_reward = 0
    dummy_done = False 
    
    for _ in range(3):
        agent.memory.add_experience(
            frame, dummy_reward, dummy_action, dummy_done
        )


def make_step(name, env, agent, score, debug: bool = False):
    # take action
    frame, reward, done, info = env.step(agent.memory.actions[-1])
    
    # get next state
    frame = preprocess_frame(frame)
    n_state = [
        agent.memory.frames[-3],
        agent.memory.frames[-2],
        agent.memory.frames[-1],
        frame
    ]


def main():    
    env_name = "PongDeterministic-v4"
    env = gym.make(env_name)
    frame = env.reset()
    print(frame.shape)
    print(env.action_space.n)
    print(env.unwrapped.get_action_meanings())
    print(env.action_space)
    print(env.observation_space.shape)
    
    
    nframe, reward, done, info = env.step(1)
    
    cv2.imwrite("org_frame.png", nframe)
    nframe = preprocess_frame(nframe)
    cv2.imwrite("frame.png", nframe)
    
    print(reward)
    print(done)
    
    env.close()


if __name__ == "__main__":
    main()