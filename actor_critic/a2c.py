import os
from datetime import date
import shutil
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from models import ActorCritic, Actor, Critic, ConvActorCritic


class Params:
    NUM_EPOCHS = 5000
    LR = 0.001
    BATCH_SIZE = 64
    GAMMA = 0.99
    HIDDEN_SIZE = 64
    BETA = 0.1 # entropy bonus multiplier
    VALUE_LOSS_COEF = 0.5
    NSTEPS = 6


class A2C:
    def __init__(self, env_name, shared_backbone=False):
        self.num_epochs = Params.NUM_EPOCHS
        self.lr = Params.LR
        self.batch_size = Params.BATCH_SIZE
        self.gamma = Params.GAMMA
        self.hidden = Params.HIDDEN_SIZE
        self.beta = Params.BETA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.value_loss_coef = Params.VALUE_LOSS_COEF
        self.env = gym.make(env_name)
        self.shared_backbone = shared_backbone
        self.nsteps = Params.NSTEPS

        if not shared_backbone:
            self.actor = Actor(self.env.observation_space.shape[0],
                self.env.action_space.n, self.hidden)
            self.critic = Critic(self.env.observation_space.shape[0], self.hidden)
            self.optimizer_actor = optim.Adam(self.actor.parameters(), self.lr)
            self.optimizer_critic = optim.Adam(self.critic.parameters(), self.lr)
        else:
            self.model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.n, self.hidden)
            self.optimizer = optim.Adam(self.model.parameters(), self.lr)

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

                if self.shared_backbone:
                    actor_loss, entropy = self.calculate_loss(epoch_logits, epoch_weight_log_probs)
                    critic_loss = torch.mean(epoch_value_estimate_errors)
                    loss = actor_loss + self.value_loss_coef * critic_loss
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    # actor training
                    loss, entropy = self.calculate_loss(epoch_logits, epoch_weight_log_probs)
                    loss.backward()
                    self.optimizer_actor.step()
                    self.optimizer_actor.zero_grad()

                    # critic training
                    critic_loss = torch.mean(epoch_value_estimate_errors)
                    critic_loss.backward()
                    self.optimizer_critic.step()
                    self.optimizer_critic.zero_grad()

                print(f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}")

                epoch_value_estimate_errors = torch.empty(size=(0,), device=self.device)
                epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.device)
                epoch_weight_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.device)

                if np.mean(self.total_rewards) > 195.0 or epoch > self.num_epochs:
                    print("Environment solved!")
                    break


    def play_episode(self, episode):
        state = self.env.reset()

        actions = torch.empty(size=(0,), dtype=torch.long, device=self.device)
        logits = torch.empty(size=(0, self.env.action_space.n), device=self.device)
        avg_rewards = np.empty(shape=(0,), dtype=float)
        ep_rewards = np.empty(shape=(0,), dtype=float)

        # for critic
        value_estimates = torch.empty(size=(0,), device=self.device)
        states = torch.empty(size=(0, self.env.observation_space.shape[0]), dtype=float, device=self.device)

        while True:
            if self.shared_backbone:
                action_logits, value_estimate = self.model(torch.tensor(state).float().unsqueeze(0).to(self.device))
            else:
                action_logits = self.actor(torch.tensor(state).float().unsqueeze(0).to(self.device))

            logits = torch.cat((logits, action_logits), dim=0)
            action = Categorical(logits=action_logits).sample()
            actions = torch.cat((actions, action), dim=0)
            states = torch.cat((states, torch.tensor(state).float().unsqueeze(0)), dim=0)

            next_state, reward, done, _ = self.env.step(action.item())

            ep_rewards = np.concatenate((ep_rewards, np.array([reward])), axis=0)

            # create baseline with value estimation neural net
            # use value network to create a state value estimate and append
            if not self.shared_backbone:
                value_estimate = self.critic(
                        torch.tensor(next_state).float().unsqueeze(0).to(self.device))

            value_estimates = torch.cat((value_estimates, value_estimate), dim=0)
            state = next_state

            if done:
                episode += 1

                # these should be the state values
                V_tar_nsteps = self.get_discounted_rewards(ep_rewards, self.gamma, states, self.nsteps)
                V_tar_nsteps = torch.tensor(V_tar_nsteps).float().to(self.device)

                value_estimation_error = torch.sum(
                        torch.pow((value_estimates - V_tar_nsteps), 2.0)).unsqueeze(0)
                # nsteps advantage calculation
                advantage = V_tar_nsteps.detach().cpu().numpy() - value_estimates.detach().cpu().numpy()

                sum_of_rewards = np.sum(ep_rewards)

                mask = F.one_hot(actions, num_classes=self.env.action_space.n)

                ep_log_probs = torch.sum(mask.float() * F.log_softmax(logits, dim=1), dim=1)

                weighted_log_probs = ep_log_probs * torch.tensor(advantage).float().to(self.device)

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


    # monte carlo estimate for V_tar
    def get_discounted_rewards(self, rewards, gamma, states=None, nsteps=1):
        V_tar = np.empty_like(rewards, dtype=float)
        if nsteps == 1:
            for i in range(rewards.shape[0]):
                gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
                discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
                V_tar_val = np.sum(rewards[i:] * discounted_gammas)
                V_tar[i] = V_tar_val

        # nstep returns
        else:
            for i in range(rewards.shape[0]):
                if i < rewards.shape[0] - nsteps - 1:
                    gammas = np.full(shape=(rewards[i:i + nsteps + 1].shape[0]), fill_value=gamma)
                    discounted_gammas = np.power(gammas, np.arange(rewards[i: i + nsteps + 1].shape[0]))
                    V_tar_val = np.sum(rewards[i: i + nsteps] * discounted_gammas[:-1])
                    with torch.no_grad():
                        if not self.shared_backbone:
                            V_estimate = self.critic(states[i + nsteps + 1].float().unsqueeze(0)).detach().item()
                            V_tar_val += V_estimate * discounted_gammas[-1]
                        else:
                            raise NotImplementedError

                else:
                    gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
                    discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
                    V_tar_val = np.sum(rewards[i:] * discounted_gammas)

                V_tar[i] = V_tar_val

            return V_tar


def main():
    shared_backbone = False
    env_name = "CartPole-v1"
    day = date.today().strftime("%Y-%m-%d")
    exp_dir = os.path.join(os.getcwd(), "exps", day + "-" + env_name)
    # if experiment directory exists, remove it
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)
    else:
        os.makedirs(exp_dir)

    a2c = A2C(env_name, shared_backbone)
    a2c.solve_env()

    if shared_backbone:
        torch.save(a2c.model.state_dict(), os.path.join(exp_dir, "a2c.pth"))
    else:
        torch.save(a2c.actor.state_dict(), os.path.join(exp_dir, "actor.pth"))
        torch.save(a2c.critic.state_dict(), os.path.join(exp_dir, "critic.pth"))


if __name__ == "__main__":
    main()

