import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window=100):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_learning_curve(rewards, label="Agent", window=100, save_path=None):
    avg_rewards = moving_average(rewards, window)

    episodes = np.arange(window - 1, len(rewards))

    plt.figure()
    plt.plot(episodes, avg_rewards, label=label)

    plt.xlabel("Episode")
    plt.ylabel(f"Average Return ({window}-episode window)")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def compare_learning_curves(results, window=100, save_path=None):
    """
    results example:

    {
        "Q-learning": q_learning_rewards,
        "SARSA": sarsa_rewards
    }
    """

    plt.figure()

    for label, rewards in results.items():
        avg_rewards = moving_average(rewards, window)
        episodes = np.arange(window - 1, len(rewards))

        plt.plot(episodes, avg_rewards, label=label)

    plt.xlabel("Episode")
    plt.ylabel(f"Average Return ({window}-episode window)")
    plt.title("Learning Curve Comparison")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_episode_length(episode_lengths, label="Agent", window=100, save_path=None):
    avg_lengths = moving_average(episode_lengths, window)

    episodes = np.arange(window - 1, len(episode_lengths))

    plt.figure()
    plt.plot(episodes, avg_lengths, label=label)

    plt.xlabel("Episode")
    plt.ylabel(f"Average Episode Length ({window}-episode window)")
    plt.title("Episode Length During Training")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def print_policy(env, agent):
    action_symbols = {
        0: "↑",
        1: "→",
        2: "↓",
        3: "←",
    }

    for row in range(env.rows):
        for col in range(env.cols):
            position = (row, col)

            if position in env.walls:
                print("#", end=" ")

            elif position == env.goal_position:
                print("G", end=" ")

            else:
                best_action = np.argmax(agent.q_table[row, col])
                print(action_symbols[best_action], end=" ")

        print()


def compare_episode_lengths(results, window=100, save_path=None):
    plt.figure()

    for label, lengths in results.items():
        avg_lengths = moving_average(lengths, window)
        episodes = np.arange(window - 1, len(lengths))

        plt.plot(episodes, avg_lengths, label=label)

    plt.xlabel("Episode")
    plt.ylabel(f"Average Episode Length ({window}-episode window)")
    plt.title("Episode Length Comparison")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
