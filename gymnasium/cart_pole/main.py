import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# q-learning -> take action -> epsilon -> next state 

env = gym.make("CartPole-v1", render_mode="rgb_array")
obs_low = np.array([-4.8, -4.0, -0.418, -4.0])
obs_high = np.array([4.8, 4.0, 0.418, 4.0])
bins = np.array([20, 15, 20, 15])
difference = obs_high - obs_low
step = difference/bins

def q_learning(alpha, gamma, epsilon, min_epsilon, decay_rate, num_episodes):
    print("Training...")
    actions = np.arange(env.action_space.n)
    
    Q = np.random.uniform(low = -1, high = 1, size = (*bins, len(actions)))
    total_reward = 0
    for i in range(num_episodes + 1):
        if i % 1000 == 0:
            print(f"Episode {i}/{num_episodes} | Average Reward (in last 1000 eps): {total_reward/1000}")
            total_reward = 0
        state, _ = env.reset()
        state = discretize_state(state)
        completed = 0
        while not completed:
            action = take_action(Q, actions, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state)
            total_reward += reward
            completed = terminated or truncated
            # bellman eqn
            Q[*state, action] += alpha * (reward + gamma * np.max(Q[*next_state]) - Q[*state, action])
            state = next_state
            epsilon = max(epsilon * decay_rate, min_epsilon) # decay epsilon
    return Q

def take_action(Q, actions, state, epsilon):
    if np.random.rand() > epsilon:
        action = np.argmax(Q[*state])
    else:
        action = np.random.randint(0, len(actions))
    return action

def discretize_state(state):
    state = ((state - obs_low)/step).astype(int)
    state = np.clip(state, 0, bins - 1)
    return state

def evaluate(Q):
    print("Evaluating...")
    completed = 0
    total_reward = 0
    state, _ = env.reset()
    state = discretize_state(state)
    while not completed:
        action = np.argmax(Q[*state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        completed = terminated or truncated
        state = discretize_state(next_state)
        frame = env.render()    # storing the frame
        plt.imshow(frame)
        plt.axis("off")
        plt.pause(0.001)    # pausing for a small time to show continuous frames
        plt.clf()    # clear last frame
    return total_reward

Q = q_learning(0.2, 0.999, 1.0, 0.05, 0.9995, 10000)
print(f"Total Reward after evaluating (1 episode): {evaluate(Q)}")