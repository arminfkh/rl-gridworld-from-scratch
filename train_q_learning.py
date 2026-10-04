from environment import GridWorld
from agents.q_learning import QLearningAgent

import statistics
import numpy as np

env = GridWorld()
agent = QLearningAgent(
    env.rows, env.cols, num_actions=4, epsilon=0.1, alpha=0.1, gamma=0.99
)

num_episodes = 5000
max_steps = 100

successes = 0
total_rewards = []
episode_lengths = []


def print_policy(env, agent):
    action_symbols = {0: "↑", 1: "→", 2: "↓", 3: "←"}

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


for episode in range(num_episodes):

    state = env.reset()
    total_reward = 0
    episode_length = 0

    for step in range(max_steps):

        action = agent.choose_action(state)
        next_state, reward, terminated = env.step(action)

        agent.update(state, action, reward, next_state, terminated)
        state = next_state

        total_reward += reward
        episode_length += 1

        if terminated:
            successes += 1
            break

    total_rewards.append(total_reward)
    episode_lengths.append(episode_length)


success_rate = successes / num_episodes
average_return = statistics.mean(total_rewards)
average_episode_length = statistics.mean(episode_lengths)

print("\nLearned Policy:")
print_policy(env, agent)

print(f"Success rate: {success_rate:.2%}")
print(f"Average return: {average_return:.2f}")
print(f"Average episode length: {average_episode_length:.2f}")
