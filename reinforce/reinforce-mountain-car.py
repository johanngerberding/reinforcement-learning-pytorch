from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def plot_rewards_losses(rewards: list, losses: list, out_dir: str = ""):
    assert len(rewards) == len(losses)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=False)
    axs[0].plot([e for e in range(1, len(rewards) + 1)], rewards, label="reward")
    axs[0].legend()
    axs[1].plot([e for e in range(1, len(losses) + 1)], losses, label="loss")
    axs[1].legend()
    plt.show()
    if out_dir != "":
        plt.savefig("rewards_losses.png")


class Pi(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()


def train(pi, optimizer, gamma: float):
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = -log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def get_reward(state):
    if state >= 0.5:
        return 10.0
    elif state > -0.4:
        return (1 + state)**3
    else:
        return 0.0


def main():
    verbose = True
    env_name = "MountainCar-v0"
    episodes = 20000
    timesteps = 250
    lr = 0.003
    gamma = 0.99
    hidden_dim = 128

    exp_rewards = []
    exp_losses = []
    highscore = 0

    env = gym.make(env_name)
    env._max_episodes_steps = 250
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    pi = Pi(in_dim, out_dim, hidden_dim)
    pi.train()
    optimizer = optim.Adam(pi.parameters(), lr=lr)

    for ep in range(episodes):
        state = env.reset()
        score = 0.0
        for t in range(timesteps):
            action = pi.act(state)
            state, reward, done, info = env.step(action)
            n_reward = get_reward(state[0])
            score += n_reward
            pi.rewards.append(n_reward)
            # if ep % 500 == 0:
                # env.render()

            if done:
                if state[0] >= 0.5:
                    print("Success!")
                break

        if score > highscore:
            print(f"New highscore: {score} - Episode {ep}")
            highscore = score

        loss = train(pi, optimizer, gamma)
        total_reward = sum(pi.rewards)
        pi.onpolicy_reset()

        if verbose and ep % 200 == 0:
            print(f"Ep. {ep}: Loss {loss}. Total reward {total_reward}.")

        exp_rewards.append(total_reward)
        exp_losses.append(loss.item())

    plot_rewards_losses(exp_rewards, exp_losses)


if __name__ == "__main__":
    main()
