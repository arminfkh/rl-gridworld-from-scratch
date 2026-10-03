from environment import GridWorld
from agents.random_agent import RandomAgent
import statistics

env = GridWorld()
agent = RandomAgent()
state = env.reset()

max_steps = 100
num_episodes = 1000

successes = 0
total_rewards = []
episode_lengths = []

for episode in range(num_episodes):

    env.reset()
    total_reward = 0
    episode_length = 0

    for step in range(max_steps):

        action = agent.choose_action()
        next_state, reward, terminated = env.step(action)

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

print(f"Success rate: {success_rate:.2%}")
print(f"Average return: {average_return:.2f}")
print(f"Average episode length: {average_episode_length:.2f}")
