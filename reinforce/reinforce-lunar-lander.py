import os
import argparse
import gym
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import Counter

from utils import plot_stats


ROOT = os.path.dirname(os.path.abspath(__file__))


class Pi(nn.Module):
    def __init__(
            self,
            observation_space: int,
            action_space: int,
            hidden_dim: int = 128,
            gamma: float = 0.99,
    ):
        super(Pi, self).__init__()
        self.data = []
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(self.observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_space)
        ]
        self.model = nn.Sequential(*layers)
        self.gamma = gamma

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=0)

    def put_data(self, item):
        self.data.append(item)

    def train(self, optimizer, device):
        R = 0.0
        optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob).to(device) * R
            loss.backward()

        optimizer.step()
        self.data = []
        return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        help="Output directory for model checkpoint and stats.",
        default=os.path.join(ROOT, "exps"),
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate.",
        default=0.003,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Gamma hyperparameter",
        default=0.99,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of episodes for training.",
        default=2000,
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="Number of runs.",
        default=10,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Model checkpoint",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        help="Hidden size in the middle of Neural Net.",
        default=128,
    )

    args = parser.parse_args()

    folder = datetime.date.today().strftime("%Y-%m-%d")
    exp_dir = os.path.join(args.out, folder)
    os.makedirs(exp_dir)

    rewards = []
    losses = []
    actions = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    for r in range(1, args.runs):

        env = gym.make('LunarLander-v2')

        agent = Pi(
            env.observation_space.shape[0],
            env.action_space.n,
            args.hidden,
            args.gamma,
        )

        if args.ckpt:
            agent.load_state_dict(torch.load(args.ckpt))

        if not os.path.isdir(args.out):
            os.makedirs(args.out)

        optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=args.lr
        )

        scores = []
        loss = []
        acts = []
        highscore = 0.0

        for ep in range(args.episodes):
            obs = env.reset()
            done = False
            score = 0.0

            while not done:
                prob = agent(torch.from_numpy(obs).float().to(device))
                m = Categorical(prob)
                action = m.sample()
                obs, reward, done, info = env.step(action.item())
                # print("Reward: {}".format(reward))
                agent.put_data((reward, prob[action]))
                score += reward
                acts.append(action.item())

                if ep % 500 == 0:
                    env.render()

            if score > highscore:
                highscore = score
                print(f"********** New highscore: {score} **********")

            if done:
                scores.append(score)
                # print("episode {}, score: {}".format(ep, score))

            loss.append(agent.train(optimizer, device))

        print("Training finished!")

        outmodel = os.path.join(exp_dir, "agent_{}.pth".format(str(r)))
        torch.save(agent.state_dict(), outmodel)
        print("Agent saved: {}".format(outmodel))

        action_count = Counter(acts)

        actions.append(action_count)
        rewards.append(scores)
        losses.append(loss)

        env.close()

    plot_stats(rewards, losses, actions, exp_dir)


if __name__ == "__main__":
    main()
