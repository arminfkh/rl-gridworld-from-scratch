import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window=100):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_learning_curve(rewards, label="Agent", window=100):
    avg_rewards = moving_average(rewards, window)

    episodes = np.arange(window - 1, len(rewards))

    plt.figure()
    plt.plot(episodes, avg_rewards, label=label)

    plt.xlabel("Episode")
    plt.ylabel(f"Average Return ({window}-episode window)")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()

    plt.show()


def compare_learning_curves(results, window=100):
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

    plt.show()


def plot_episode_length(episode_lengths, label="Agent", window=100):
    avg_lengths = moving_average(episode_lengths, window)

    episodes = np.arange(window - 1, len(episode_lengths))

    plt.figure()
    plt.plot(episodes, avg_lengths, label=label)

    plt.xlabel("Episode")
    plt.ylabel(f"Average Episode Length ({window}-episode window)")
    plt.title("Episode Length During Training")
    plt.legend()
    plt.grid()

    plt.show()
