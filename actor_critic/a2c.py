import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque


class Params:
    NUM_EPOCHS = 5000
    LR = 0.0001
    BATCH_SIZE = 64
    GAMMA = 0.99
    HIDDEN_SIZE = 64
    BETA = 0.1 # entropy bonus multiplier


class Critic(nn.Module):
    def __init__(self, observation_space, hidden_size):
        super(Critic, self).__init__()
        self.model= nn.Sequential(nn.Linear(observation_space, hidden_size, bias=True),
                nn.ReLU(), nn.Linear(hidden_size, hidden_size, bias=True),
                nn.ReLU(), nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.model(x)


class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super(Actor, self).__init__()
        self.model= nn.Sequential(
                nn.Linear(observation_space, hidden_size, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_size, action_space, bias=True),
        )

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.model(x)


class PolicyGradient:
    def __init__(self):
        self.num_epochs = 5000
        self.lr = 0.00001
        self.batch_size = 32
        self.gamma = 0.99
        self.hidden = 128
        self.beta = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = gym.make('CartPole-v1')
        self.actor = Actor(self.env.observation_space.shape[0],
                self.env.action_space.n, self.hidden)
        self.critic = Critic(self.env.observation_space.shape[0], self.hidden)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), self.lr)
        self.total_rewards = deque([], maxlen=100)


    def solve_env(self):
        episode = 0
        epoch = 0

        epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.device)
        epoch_weight_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.device)

        epoch_value_estimate_errors = torch.empty(size=(0,), device=self.device)

        while True:
            # play episode
            (ep_weighted_log_prob_trajectory,
                    ep_logits, sum_of_ep_rewards, episode, sum_value_error) = self.play_episode(episode)

            self.total_rewards.append(sum_of_ep_rewards)
            epoch_weight_log_probs = torch.cat((epoch_weight_log_probs, ep_weighted_log_prob_trajectory), dim=0)
            epoch_logits = torch.cat((epoch_logits, ep_logits), dim=0)

            epoch_value_estimate_errors = torch.cat((epoch_value_estimate_errors, sum_value_error), dim=0)

            if episode >= self.batch_size:
                episode = 0
                epoch += 1
                # actor training
                loss, entropy = self.calculate_loss(epoch_logits, epoch_weight_log_probs)
                self.optimizer_actor.zero_grad()
                loss.backward()
                self.optimizer_actor.step()

                # critic training ???????
                self.optimizer_critic.zero_grad()
                critic_loss = torch.mean(epoch_value_estimate_errors)
                # print(f"Epoch value estimation error: {critic_loss.item()}")
                critic_loss.backward()
                self.optimizer_critic.step()

                print(f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}")

                epoch_value_estimate_errors = torch.empty(size=(0,), device=self.device)
                epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.device)
                epoch_weight_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.device)

                if np.mean(self.total_rewards) > 195.0:
                    print("Environment solved!")
                    break


    def play_episode(self, episode):
        state = self.env.reset()

        actions = torch.empty(size=(0,), dtype=torch.long, device=self.device)
        logits = torch.empty(size=(0, self.env.action_space.n), device=self.device)
        avg_rewards = np.empty(shape=(0,), dtype=np.float)
        ep_rewards = np.empty(shape=(0,), dtype=np.float)

        # for training the critic
        states = torch.empty(size=(0,), device=self.device)
        # for critic
        value_estimates = torch.empty(size=(0,), device=self.device)


        while True:
            action_logits = self.actor(torch.tensor(state).float().unsqueeze(0).to(self.device))
            logits = torch.cat((logits, action_logits), dim=0)
            action = Categorical(logits=action_logits).sample()
            actions = torch.cat((actions, action), dim=0)

            state, reward, done, _ = self.env.step(action.item())

            states = torch.cat((states, torch.tensor(state).float()), dim=0)
            ep_rewards = np.concatenate((ep_rewards, np.array([reward])), axis=0)

            # create baseline with value estimation neural net
            # use value network to create a state value estimate and append
            value_estimate = self.critic(torch.tensor(state).float().unsqueeze(0).to(self.device))
            value_estimates = torch.cat((value_estimates, value_estimate), dim=0)

            # avg_rewards = np.concatenate((avg_rewards,
                # np.expand_dims(np.mean(ep_rewards), axis=0)), axis=0)

            if done:
                episode += 1
                discounted_rewards_to_go = PolicyGradient.get_discounted_rewards(
                        ep_rewards, self.gamma)

                # these should be the state values
                return_values = PolicyGradient.get_discounted_rewards(ep_rewards, 1.0)
                return_values = torch.tensor(return_values).float().to(self.device)
                value_estimation_error = torch.sum(torch.pow((value_estimates - return_values), 2.0)).unsqueeze(0)
                # value_estimation_error = 1 / value_estimates.shape[0] * value_estimation_error
                # print(f"Value estimation error: {value_estimation_error}")
                # subtract baseline
                # discounted_rewards_to_go -= avg_rewards
                baseline = value_estimates.detach().cpu().numpy().squeeze()

                discounted_rewards_to_go -= baseline
                sum_of_rewards = np.sum(ep_rewards)

                mask = F.one_hot(actions, num_classes=self.env.action_space.n)

                ep_log_probs = torch.sum(mask.float() * F.log_softmax(logits, dim=1), dim=1)

                weighted_log_probs = ep_log_probs * torch.tensor(discounted_rewards_to_go).float().to(self.device)

                sum_weighted_log_probs = torch.sum(weighted_log_probs).unsqueeze(0)

                return sum_weighted_log_probs, logits, sum_of_rewards, episode, value_estimation_error


    def calculate_loss(self, epoch_logits, weighted_log_probs, entropy_bonus=True):
        loss = -1 * torch.mean(weighted_log_probs)
        entropy = None
        entropy_bonus = 0.0

        if entropy_bonus:
            # entropy bonus
            p = F.softmax(epoch_logits, dim=1)
            log_p = F.log_softmax(epoch_logits, dim=1)
            entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
            entropy_bonus = -1 * self.beta * entropy

        return loss + entropy_bonus, entropy



    @staticmethod
    def get_discounted_rewards(rewards, gamma):
        discounted_rewards = np.empty_like(rewards, dtype=np.float)
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward

        return discounted_rewards



def main():

    exp_dir = "exps"
    pg = PolicyGradient()
    pg.solve_env()
    torch.save(pg.actor.state_dict(), os.path.join(exp_dir, "actor.pth"))
    torch.save(pg.critic.state_dict(), os.path.join(exp_dir, "critic.pth"))



if __name__ == "__main__":
    main()

