import gym 
import cv2 
import numpy as np 
from collections import deque

# environment wrappers from here:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        "Take action on reset for environments that are fixed until firing."
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class MaxAndSkip(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        "Return only every `skip`-th frame."
        super(MaxAndSkip, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip 
    
    def step(self, action):
        total_reward = 0.0
        done = None 
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break 
        
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs 
    

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        "Warp frames like in the paper."
        super().__init__(env)
        self._width = width
        self._height = height 
        self._grayscale = grayscale 
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3
            
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        ) 
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else: 
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.space[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3
        
    def observation(self, obs):
        if self._key is None:
            frame = obs 
        else: 
            frame = obs[self._key]
        
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame = cv2.resize(
            frame, 
            (self._width, self._height), 
            interpolation=cv2.INTER_AREA,
        )
        
        if self._grayscale:
            frame = np.expand_dims(frame, -1)
        
        if self._key is None:
            obs = frame 
        else:
            obs = obs.copy()
            obs[self._key] = frame 
        
        return obs 


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        "Reward => (-1, 0, +1)"
        gym.RewardWrapper.__init__(self, env)
    
    def reward(self, reward):
        return np.sign(reward)
    
    
class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        "Stack k last frames."
        super(FrameStack, self).__init__(env)
        self.k = k 
        self.frames = deque([], maxlen=k)
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[:-1] + (old_shape[-1] * k,)),
            dtype=env.observation_space.dtype,
        )
    
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info 
    
    def _get_obs(self):
        assert len(self.frames) == self.k 
        # return LazyFrames(list(self.frames))
        return np.array(list(self.frames))


class ImageToPytorch(gym.ObservationWrapper):
    def __init__(self, env):
        "Change axis and normalize for PyTorch"
        super(ImageToPytorch, self).__init__(env)
        old_shape = self.observation_space.shape 
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=old_shape[:-1],
            dtype=np.float32,
        )
        
    def observation(self, obs):
        obs = np.array(obs).astype(np.float32) / 255.0
        return np.squeeze(obs)
      
    
class LazyFrames:
    def __init__(self, frames):
        "This is for memory optimization."
        self._frames = frames
        self._out = None 
        
    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out 
    
    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out 
    
    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
    
    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]
    
    def frame(self, i):
        return self._force()[..., i]
            
        
        
def generate_env(
    env_name: str, 
    skip: bool = True, 
    clip_rewards: bool = True, 
    frame_stack: bool = True, 
    pytorch: bool = True,
    num_frames: int = 4,
):
    "Generate environment including wrappers."
    env = gym.make(env_name)
    
    if skip:
        env = MaxAndSkip(env, skip=4)
    
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    
    if clip_rewards:
        env = ClipRewardEnv(env)
    
    if frame_stack and num_frames > 1:
        env = FrameStack(env, num_frames)
    
    if pytorch: 
        env = ImageToPytorch(env)
    
    return env