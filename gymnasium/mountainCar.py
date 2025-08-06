import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0")

q_table = np.zeros((18, 14,3))

alpha = 0.1             
gamma = 0.99            
epsilon = 1.0           
min_epsilon = 0.01
epsilon_decay = 0.995
episodes = 2000
max_steps = 200

def discretize(obs):
    pos, vel = obs

    pos_scaled = (pos + 1.2) / 1.8
    vel_scaled = (vel + 0.07) / 0.14

    pos_car = int(pos_scaled * (17))
    vel_car = int(vel_scaled * (13))

    pos_car = np.clip(pos_car, 0, 18)
    vel_car = np.clip(vel_car, 0, 13)

    return (pos_car, vel_car)

for episode in range(episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    total_reward = 0

    for step in range(max_steps):
        # Choose action: epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, reward, done, truncated, _ = env.step(action)
        next_state = discretize(next_obs)

        # Q-learning update 
        best_next_action = np.max(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * best_next_action - q_table[state][action])

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training complete ")

env=gym.make("MountainCar-v0", render_mode="human")
obs, _ = env.reset()
state = discretize(obs)
done = False

while not done:
    env.render()
    action = np.argmax(q_table[state])
    obs, reward, done, truncated, _ = env.step(action)
    state = discretize(obs)

env.close()