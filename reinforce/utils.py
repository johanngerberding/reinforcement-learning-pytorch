import os
import random
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


def get_reward_mountain_car(state):
    "Custom reward for the MountainCar environment"
    if state >= 0.5:
        return 10.0
    elif state > -0.4:
        return (1 + state)**4
    else:
        return 0.0


def plot_stats(
    env_name: str,
    rewards: List[List],
    losses: List[List],
    actions: Optional[List[List]],
    save_dir: str,
    figsize: tuple = (30, 6),
    window_size: int = 10,
    save: bool = True,
):
    assert len(rewards) == len(losses)

    if actions:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

    fig.tight_layout(pad=7.5)
    fig.suptitle("{} - {} runs".format(env_name, len(rewards)))
    nrewards = []
    nlosses = []

    if window_size > 1:
        for i in range(len(rewards)):
            nrewards.append([sum([rewards[i][j]
                                  for j in range(k, k + window_size)]) / window_size
                             for k in range(0, len(rewards[i]), window_size)])
            nlosses.append([sum([losses[i][j]
                                 for j in range(k, k + window_size)]) / window_size
                            for k in range(0, len(losses[i]), window_size)])

    mean_rewards = []
    std_rewards = []
    mean_losses = []
    std_losses = []
    mean_actions = []
    std_actions = []

    for i in range(len(nrewards[0])):
        nums = [nrewards[j][i] for j in range(len(nrewards))]
        std_rewards.append(np.std(nums))
        m = sum(nums) / len(nrewards)
        mean_rewards.append(m)

    for i in range(len(nlosses[0])):
        nums = [nlosses[j][i] for j in range(len(nlosses))]
        std_losses.append(np.std(nums))
        l = sum(nums) / len(nlosses)
        mean_losses.append(l)

    for i in range(len(actions[0])):
        nums = [actions[j][i] for j in range(len(actions))]
        std_actions.append(np.std(nums))
        a = sum(nums) / len(actions)
        mean_actions.append(a)

    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)
    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)

    with sns.axes_style("darkgrid"):
        axs[0].plot(mean_rewards, color='#006600', label="rewards")
        axs[0].fill_between([x for x in range(len(mean_rewards))],
                            mean_rewards - std_rewards,
                            mean_rewards + std_rewards,
                            alpha=0.3,
                            edgecolor='#006600',
                            facecolor='#006600')
        axs[0].set_title("Mean Rewards per Episode")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")
        axs[0].set_xticklabels(
            [int(t * window_size) for t in axs[0].get_xticks().tolist()]
        )

        axs[1].plot(mean_losses, color='#cc0000', label="losses")
        axs[1].fill_between([x for x in range(len(mean_losses))],
                            mean_losses - std_losses,
                            mean_losses + std_losses,
                            alpha=0.3,
                            edgecolor='#cc0000',
                            facecolor='#cc0000')
        axs[1].set_title("Mean Loss per Episode")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Loss")
        axs[1].set_xticklabels(
            [int(t * window_size) for t in axs[1].get_xticks().tolist()]
        )

        if actions:
            axs[2].bar([str(i) for i in range(1, len(mean_actions) + 1)],
                       np.array(mean_actions) / sum(mean_actions), color='#0033cc')
            axs[2].set_title("Mean Action Selection")
            axs[2].set_xlabel("Action")
            axs[2].set_yticklabels(
                ["{}%".format(int(t * 100)) for t in axs[2].get_yticks().tolist()]
            )

        if save:
            out = os.path.join(save_dir, "results.png")
            fig.savefig(out)
        
        plt.show()


# TESTING
def main():
    test_reward = [[random.randint(0, 200) for _ in range(1000)]
                   for _ in range(10)]
    test_loss = [[random.randint(0, 400) for _ in range(1000)]
                 for _ in range(10)]
    test_action = [[random.randint(5, 50) for _ in range(5)]
                   for _ in range(10)]

    root = os.path.dirname(os.path.abspath(__file__))
    plot_stats("TEST_ENV", test_reward, test_loss, test_action, root)


if __name__ == "__main__":
    main()
