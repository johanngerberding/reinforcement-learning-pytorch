import os
import random
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


def plot_stats(
    rewards: List[List],
    losses: List[List],
    actions: Optional[List[List]],
    save_dir: str,
    figsize: tuple = (30, 8),
    window_size: int = 10,
    save: bool = True,
):
    assert len(rewards) == len(losses)

    if actions:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

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
        axs[0].set_xlabel("Episode (in {})".format(window_size))
        axs[0].set_ylabel("Reward")
        axs[1].plot(mean_losses, color='#cc0000', label="losses")
        axs[1].fill_between([x for x in range(len(mean_losses))],
                            mean_losses - std_losses,
                            mean_losses + std_losses,
                            alpha=0.3,
                            edgecolor='#cc0000',
                            facecolor='#cc0000')
        axs[1].set_title("Mean Loss per Episode")
        axs[1].set_xlabel("Episode (in {})".format(window_size))
        axs[1].set_ylabel("Loss")

        if actions:
            axs[2].bar([str(i) for i in range(1, len(mean_actions) + 1)],
                       mean_actions, color='#0033cc')
            axs[2].set_title("Mean Action Selection Counts")
            axs[2].set_xlabel("Action")
            axs[2].set_ylabel("Action Count")

        plt.show()

        fig.savefig("res.png")
        if save:
            out = os.path.join(save_dir, "results.png")
            fig.savefig(out)


# TESTING
def main():
    test_reward = [[random.randint(0, 200) for _ in range(1000)]
                   for _ in range(10)]
    test_loss = [[random.randint(0, 400) for _ in range(1000)]
                 for _ in range(10)]
    test_action = [[random.randint(5, 50) for _ in range(5)]
                   for _ in range(10)]

    root = os.path.dirname(os.path.abspath(__file__))
    plot_stats(test_reward, test_loss, test_action, root)


if __name__ == "__main__":
    main()
