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
    parser = argparse.ArgumentParser()  
    parser.add_argument("--env", type=str, 
                        help="Name of the gym environment", 
                        default="PongNoFrameskip-v4")
    parser.add_argument("--gamma", type=float, 
                        help="Gamma hyperparameter", 
                        default=0.99)
    parser.add_argument("--lr", type=float, 
                        help="Learning Rate", 
                        default=1e-4)
    parser.add_argument("--batch_size", type=int, 
                        help="Batch size to train Q-Network", 
                        default=32)
    parser.add_argument("--eps", type=float, 
                        help="Epsilon starting value", 
                        default=1e-4)
    parser.add_argument("--eps_min", type=float, 
                        help="Epsilon minimum value", 
                        default=0.02)
    parser.add_argument("--eps_decay", type=float, 
                        help="Epsilon decay rate", 
                        default=0.999985)
    parser.add_argument("--replay_buffer", type=int, 
                        help="Size of the replay buffer", 
                        default=100000)
    parser.add_argument("--min_exps", type=int, 
                        help="Minimum number of experiences before training", 
                        default=10000)
    parser.add_argument("--sync", type=int, 
                        help="Network synchronisation interval", 
                        default=1000)
    parser.add_argument("--reward_bound", type=float, 
                        help="Bound for reward to be done", 
                        default=21.0)
    parser.add_argument("--out", type=str, 
                        help="Path to output directory", default=None)
    args = parser.parse_args()
    
    if not args.out:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        d = date.today().strftime("%d-%m-%Y")
        outdir = os.path.join(curr_dir, args.env + "_" + d)
    else: 
        outdir = args.out 
    
    os.makedirs(outdir, exist_ok=False)
    
    # Save hyperparameters 
    with open(os.path.join(outdir, "params.json"), 'wt') as f:
        json.dump(vars(args), f, indent=4)
    
    env = generate_env(args.env)
    _ = env.reset()
    
    Q_network = DQN_Conv(4, env.action_space.n).to(DEVICE)
    Q_tar_network = DQN_Conv(4, env.action_space.n).to(DEVICE)
    
    replay_buffer = ReplayBuffer(args.replay_buffer)
    agent = Agent(env, replay_buffer)
    
    writer = SummaryWriter(
        log_dir=outdir,
        comment="-" + args.env
    )
    
    eps = args.eps
    
    optimizer = torch.optim.Adam(Q_network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    
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
        
        # Training Phase
        optimizer.zero_grad()
        batch = replay_buffer.sample(args.batch_size)
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
        
        expected_state_action_values = next_state_values * args.gamma + rewards_t 
        loss_t = torch.nn.MSELoss()(state_action_values, expected_state_action_values)
        
        loss_t.backward()
        scheduler.step(loss_t) 
        
    writer.close()
    env.close()
    end = time.time()
    training_time = end - start
    h = training_time // 3600
    min = training_time % 3600 // 60
    print(f"Training took {int(h)} hours and {int(min)} minutes.")


if __name__ == "__main__":
    main()