from train_q_learning import train_q_learning
from train_sarsa import train_sarsa
from utils.visualization import (
    compare_learning_curves,
    compare_episode_lengths,
)

import statistics
import numpy as np

NUM_EPISODES = 5000
MAX_STEPS = 100
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.99


# Q-learning
q_agent, q_rewards, q_lengths, q_success_rate = train_q_learning(
    num_episodes=NUM_EPISODES,
    max_steps=MAX_STEPS,
    epsilon=EPSILON,
    alpha=ALPHA,
    gamma=GAMMA,
)


# SARSA
sarsa_agent, sarsa_rewards, sarsa_lengths, sarsa_success_rate = train_sarsa(
    num_episodes=NUM_EPISODES,
    max_steps=MAX_STEPS,
    epsilon=EPSILON,
    alpha=ALPHA,
    gamma=GAMMA,
)


# Summary statistics
q_learning_average_return = statistics.mean(q_rewards)
q_learning_average_length = statistics.mean(q_lengths)

sarsa_average_return = statistics.mean(sarsa_rewards)
sarsa_average_length = statistics.mean(sarsa_lengths)


print("\nAlgorithm Comparison")

print("\nQ-learning")
print(f"Success rate: {q_success_rate:.2%}")
print(f"Average return: {q_learning_average_return:.2f}")
print(f"Average episode length: {q_learning_average_length:.2f}")

print("\nSARSA")
print(f"Success rate: {sarsa_success_rate:.2%}")
print(f"Average return: {sarsa_average_return:.2f}")
print(f"Average episode length: {sarsa_average_length:.2f}")


compare_learning_curves(
    {
        "Q-learning": q_rewards,
        "SARSA": sarsa_rewards,
    },
    window=100,
    save_path="plots/comparison/return_comparison.png",
)

compare_episode_lengths(
    {
        "Q-learning": q_lengths,
        "SARSA": sarsa_lengths,
    },
    window=100,
    save_path="plots/comparison/episode_length_comparison.png",
)
