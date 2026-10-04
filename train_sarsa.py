from environment import GridWorld
from agents.sarsa import SarsaAgent
from utils.visualization import plot_learning_curve, plot_episode_length, print_policy

import statistics
import numpy as np

env = GridWorld()

agent = SarsaAgent(
    env.rows, env.cols, num_actions=4, epsilon=0.1, alpha=0.1, gamma=0.99
)

num_episodes = 5000
max_steps = 100

successes = 0
total_rewards = []
episode_lengths = []

for episode in range(num_episodes):

    state = env.reset()
    action = agent.choose_action(state)

    total_reward = 0
    episode_length = 0

    for step in range(max_steps):

        next_state, reward, terminated = env.step(action)

        if terminated:
            next_action = None
        else:
            next_action = agent.choose_action(next_state)

        agent.update(state, action, reward, next_state, next_action, terminated)

        total_reward += reward
        episode_length += 1

        if terminated:
            successes += 1
            break

        state = next_state
        action = next_action

    total_rewards.append(total_reward)
    episode_lengths.append(episode_length)

success_rate = successes / num_episodes
average_return = statistics.mean(total_rewards)
average_episode_length = statistics.mean(episode_lengths)

print("\nSARSA Training")
print(f"Success rate: {success_rate:.2%}")
print(f"Average return: {average_return:.2f}")
print(f"Average episode length: {average_episode_length:.2f}")


agent.epsilon = 0

state = env.reset()
total_reward = 0

for step in range(max_steps):

    action = agent.choose_action(state)
    next_state, reward, terminated = env.step(action)

    state = next_state

    total_reward += reward

    if terminated:
        break

print("\nSARSA Evaluation")
print(f"Greedy policy return: {total_reward}")
print(f"Greedy policy length: {step + 1}")
print(f"Reached goal: {terminated}")

print("\nLearned Policy:")
print_policy(env, agent)

plot_learning_curve(total_rewards, label="SARSA", window=100)

plot_episode_length(episode_lengths, label="SARSA", window=100)
