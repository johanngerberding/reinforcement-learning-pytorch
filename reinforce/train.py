import os
import datetime
import argparse
import gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from collections import Counter

from reinforce import Pi
from utils import plot_stats, get_reward_mountain_car


ROOT = os.path.dirname(os.path.abspath(__file__))
ENVS = [
    "MountainCar-v0",
    "LunarLander-v2",
    "CartPole-v0"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name.",
        default="LunarLander-v2",
    )
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
        default=3500,
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
    parser.add_argument(
        "--env_max_epi_steps",
        type=int,
        help="Modified max environment steps.",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        help="Print interval.",
        default=500,
    )
    parser.add_argument(
        "--render",
        type=int,
        help="Rendering interval",
    )

    args = parser.parse_args()

    assert args.env in ENVS, "Choose one of the following environments: {}".format(ENVS)
    folder = datetime.date.today().strftime("%Y-%m-%d")
    exp_dir = os.path.join(args.out, folder + "_" + args.env)
    while os.path.isdir(exp_dir):
        exp_dir = exp_dir + "_"
    os.makedirs(exp_dir)

    rewards = []
    losses = []
    actions = []

    print("START TRAINING:")
    print("-"*50)
    print("environment: {}".format(args.env))
    print("# runs: {}".format(args.runs))
    print("# episodes: {}".format(args.episodes))
    print("learning rate: {}".format(args.lr))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))
    print("-"*50)

    for r in range(1, args.runs + 1):

        scores = []
        exp_losses = []
        exp_actions = []
        highscore = 0

        env = gym.make(args.env)
        if args.env_max_epi_steps:
            env._max_episodes_steps = args.env_max_epi_steps

        pi = Pi(
            env.observation_space.shape[0],
            env.action_space.n,
            args.hidden,
            args.gamma,
        ).to(device)

        optimizer = optim.Adam(pi.parameters(), lr=args.lr)

        for ep in range(args.episodes):
            state = env.reset()
            score = 0.0
            done = False
            while not done:
                prob = pi(torch.from_numpy(state).float().to(device))
                m = Categorical(prob)
                action = m.sample()
                state, reward, done, _ = env.step(action.item())

                if args.env == "MountainCar-v0":
                    n_reward = get_reward_mountain_car(state[0])
                    score += n_reward
                    pi.put_data((n_reward, prob[action]))
                else:
                    score += reward
                    pi.put_data((reward, prob[action]))

                exp_actions.append(action.item())

                if args.render and ep % args.render == 0:
                    env.render()

                if done:
                    scores.append(score)

                if done and args.env == "MountainCar-v0":
                    if state[0] >= 0.5:
                        print("Success!")
                    break

            exp_losses.append(pi.train(optimizer, device))

            if score > highscore:
                print(f"New highscore: {score} - Episode {ep}")
                highscore = score

            if args.print_interval and ep % args.print_interval == 0:
                print(f"Run: {r} - Ep. {ep}: Loss {exp_losses[-1]}. Total reward {scores[-1]}.")


        print("Training finished")
        outmodel = os.path.join(exp_dir, "agent_{}.pth".format(str(r)))
        torch.save(pi.state_dict(), outmodel)
        print("Agent saved: {}".format(outmodel))

        action_counts = Counter(exp_actions)
        rewards.append(scores)
        losses.append(exp_losses)
        actions.append(action_counts)
        env.close()

    plot_stats(args.env, rewards, losses, actions, exp_dir)


if __name__ == "__main__":
    main()
