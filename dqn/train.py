import os 
import argparse 
import torch
import json 
import time 
import gym.spaces 
from collections import deque, namedtuple
import numpy as np 
from datetime import date
from torch.utils.tensorboard import SummaryWriter

from atari_wrappers import generate_env
from dqn import DQN, DQN_Conv, NoisyDQN

# this helps for debugging
torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple(
    "Experience", field_names=[
        "state", "action", "reward", "done", "next_state",
    ]
)


class ReplayBuffer:
    def __init__(self, capacity: int, gamma: float, nstep: int = 1):
        self.nstep = nstep
        self.gamma = gamma
        self.buffer = deque(maxlen=capacity)
        self.nstep_buffer = deque(maxlen=nstep)
        
    def __len__(self):
        return len(self.buffer)

    def _get_nsteps(self):
        reward, done, next_state = self.nstep_buffer[-1][-3:]
        for _, _, rew, do, next_st in reversed(list(self.nstep_buffer)[:-1]):
            reward = rew + self.gamma * reward * (1 - do)
            next_state, done = (next_st, do) if do else (next_state, done) 
        return reward, done, next_state

    def append(self, experience: Experience):
        self.nstep_buffer.append(experience)
        # if we haven't enough samples, wait till buffer full       
        if len(self.nstep_buffer) < self.nstep:
            return 
        reward, done, next_state = self._get_nsteps()
        state, action = self.nstep_buffer[0][:2]
        exp = Experience(state, action, reward, done, next_state)
        self.buffer.append(exp)
    
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
    def __init__(self, env, replay_buffer, noisy=False):
        self.env = env
        self.replay_buffer = replay_buffer
        self.noisy = noisy
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0 

    #@torch.no_grad()
    def step(self, model, epsilon=0.0):
        done_reward = None
        
        if np.random.random() < epsilon and not self.noisy:
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

# Tested environments
ATARI_ENVS = ["PongNoFrameskip-v4"]
OTHER_ENVS = ["CartPole-v0", "MountainCar-v0"]
                  
        
def main():  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--env", type=str, 
                        help="Name of the gym environment", 
                        default="MountainCar-v0")
    parser.add_argument("--gamma", type=float, 
                        help="Gamma hyperparameter", 
                        default=1.0)
    parser.add_argument("--lr", type=float, 
                        help="Learning Rate", 
                        default=1e-4)
    parser.add_argument("--batch_size", type=int, 
                        help="Batch size to train Q-Network", 
                        default=32)
    parser.add_argument("--eps", type=float, 
                        help="Epsilon starting value", 
                        default=1.0)
    parser.add_argument("--eps_min", type=float, 
                        help="Epsilon minimum value", 
                        default=0.01)
    parser.add_argument("--eps_decay", type=float, 
                        help="Epsilon decay rate", 
                        default=0.99985)
    parser.add_argument("--replay_buffer", type=int, 
                        help="Size of the replay buffer", 
                        default=10000)
    parser.add_argument("--min_exps", type=int, 
                        help="Minimum number of experiences before training", 
                        default=1000)
    parser.add_argument("--sync", type=int, 
                        help="Network synchronisation interval", 
                        default=1500)
    parser.add_argument("--reward_bound", type=float, 
                        help="Bound for reward to be done", 
                        default=0.5)
    parser.add_argument("--nstep", type=int, 
                        help="n-step DQN", 
                        default=3)
    parser.add_argument("--ddqn", type=bool, 
                        help="Double DQN", 
                        default=False)
    parser.add_argument("--noisy", type=str, 
                        help="Noisy networks, options: ['independent', 'factorized']")
    parser.add_argument("--out", type=str, 
                        help="Path to output directory", default=None)
    args = parser.parse_args()
    
    assert args.noisy in ['independent', 'factorized', None]
    assert args.env in ATARI_ENVS or args.env in OTHER_ENVS
    
    if not args.out:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        d = date.today().strftime("%d-%m-%Y")
        outdir = os.path.join(curr_dir, "exps", args.env + "_" + d)
    else: 
        outdir = args.out 
    
    os.makedirs(outdir, exist_ok=False)
    
    # Save hyperparameters 
    with open(os.path.join(outdir, "params.json"), 'wt') as f:
        json.dump(vars(args), f, indent=4)
    
    if args.env in ATARI_ENVS:
        env = generate_env(args.env)
    else: 
        env = gym.make(args.env)
        
    frame = env.reset()
       
    if args.noisy:
        Q_network = NoisyDQN(
            frame.shape[0], 
            frame.shape[1:], 
            env.action_space.n
        ).to(DEVICE)
        Q_tar_network = NoisyDQN(
            frame.shape[0], 
            frame.shape[1:], 
            env.action_space.n
        ).to(DEVICE)
    elif args.env in ATARI_ENVS:
        Q_network = DQN_Conv(frame.shape[0], env.action_space.n).to(DEVICE)
        Q_tar_network = DQN_Conv(frame.shape[0], env.action_space.n).to(DEVICE)
    else: 
        Q_network = DQN(env.observation_space.shape[0], env.action_space.n, args.gamma).to(DEVICE)
        Q_tar_network = DQN(env.observation_space.shape[0], env.action_space.n, args.gamma).to(DEVICE)
    
    replay_buffer = ReplayBuffer(args.replay_buffer, args.gamma, args.nstep)
    
    if args.noisy:
        agent = Agent(env, replay_buffer, True)
    else: 
        agent = Agent(env, replay_buffer)
    
    writer = SummaryWriter(
        log_dir=outdir,
        comment="-" + args.env
    )
    
    eps = args.eps
    
    optimizer = torch.optim.RMSprop(Q_network.parameters(), lr=args.lr)
    
    total_rewards = []
    frame_idx = 0 
    best_mean_reward = None 
    
    start = time.time()
    
    while True: 
        # Sampling Phase
        frame_idx += 1 
        eps = max(eps * args.eps_decay, args.eps_min)
        reward = agent.step(Q_network, eps)
        
        if reward is not None: 
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            print("{}: {} games, mean reward {}, (epsilon {})".format(frame_idx, len(total_rewards), mean_reward, eps))

            writer.add_scalar("epsilon", eps, frame_idx)
            writer.add_scalar("mean_reward_last_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("replay_buffer_size", len(replay_buffer), frame_idx)
            
            
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(Q_network.state_dict(), 
                           os.path.join(outdir, args.env + "-best.pth"))
                best_mean_reward = mean_reward

                if best_mean_reward is not None:
                    print(f"Best mean reward updated: {best_mean_reward}")
            
            if mean_reward > args.reward_bound:
                print(f"Solved in {frame_idx} frames.")
                break
                
        if len(replay_buffer) < args.min_exps:
            continue 
        
        if frame_idx % args.sync == 0:
            Q_tar_network.load_state_dict(Q_network.state_dict())
            print("Target Network updated.")
        
        if frame_idx % 500 == 0 and args.noisy:
            snr_vals = Q_network.noisy_layers_sigma_snr()
            for layer_idx, sigma_l2 in enumerate(snr_vals):
                writer.add_scalar(f"sigma_snr_layer_{layer_idx+1}", sigma_l2, frame_idx)
        
        # Training Phase
        optimizer.zero_grad()
        batch = replay_buffer.sample(args.batch_size)
        states, actions, rewards, dones, next_states = batch 
        
        states_t = torch.tensor(states).to(DEVICE)
        actions_t = torch.tensor(actions).to(DEVICE)
        rewards_t = torch.tensor(rewards).to(DEVICE)
        dones_t = torch.ByteTensor(dones).to(DEVICE)
        next_states_t = torch.tensor(next_states).to(DEVICE)
        
        state_action_values = Q_network(states_t).gather(
            1, actions_t.unsqueeze(-1)).squeeze(-1)
        
        if args.ddqn:
            next_state_actions = Q_network(next_states_t).max(1)[1]
            next_state_values = Q_tar_network(next_states_t).gather(
                1, next_state_actions.unsqueeze(-1)
            ).squeeze(-1)
        else:
            next_state_values = Q_tar_network(next_states_t).max(1)[0]
        
        next_state_values[dones_t] = 0.0 
        next_state_values = next_state_values.detach() 
        expected_state_action_values = rewards_t + (args.gamma**args.nstep) * next_state_values
       
        loss_t = torch.nn.MSELoss()(state_action_values, expected_state_action_values)
        
        loss_t.backward()
        optimizer.step() 
                
    writer.close()
    env.close()
    end = time.time()
    training_time = end - start
    h = training_time // 3600
    min = training_time % 3600 // 60
    print(f"Training took {int(h)} hours and {int(min)} minutes.")


if __name__ == "__main__":
    main()