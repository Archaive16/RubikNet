import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode="rgb_array")

# q-learning --> init state --> take action --> epsilon greedy --> update Q --> next state

def q_learning(Q, alpha, gamma, epsilon, min_epsilon, decay, num_episodes):
    print("Training...")
    actions = np.arange(0, env.action_space.n)
    i = 0
    success = 0
    for i in range(num_episodes + 1):
        if i % 1000 == 0:
            print(f"Episode {i} / {num_episodes} | Epsilon: {epsilon:.5f} | Success (in last 1000 eps) = {success}")    # printing progress
            success = 0
        state, _ = env.reset()
        completed = 0
        while not completed:
            action = take_action(actions, Q, epsilon, state)    # take action
            next_state, reward, terminated, truncated, _ = env.step(action)    # generate next state, reward and status\
            completed = terminated or truncated
            if reward == 20:
                success += 1
            # bellman equation
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            epsilon = epsilon_decay(epsilon, min_epsilon, decay)    # decay epsilon after every Q update (every step)
    return Q

def epsilon_decay(epsilon, min_epsilon, decay):
    if epsilon > min_epsilon:
        epsilon = epsilon * decay
    return epsilon

# function to take action using epsilon-greedy mechanism
def take_action(actions, Q, epsilon, state):
    if np.random.rand() > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(0, len(actions))
    return action

# evaluating final learnt Q for 1 episode
def evaluate(Q):
    print("Evaluating...")
    #plt.ion()
    state, _ = env.reset()
    total_reward = 0
    completed = 0
    while not completed:
        action = np.argmax(Q[state])    # take action
        next_state, reward, terminated, truncated, _ = env.step(action)    # generate next state, reward and status
        total_reward += reward
        completed = terminated or truncated
        state = next_state
        frame = env.render()    # storing the frame
        plt.imshow(frame)
        plt.axis("off")
        plt.pause(1)    # pausing for a small time to show continuous frames
    return total_reward

Q = np.zeros((500, 6))
Q = q_learning(Q, 0.1, 0.9999, 1.0, 0.05, 0.9995, 10000)

print(f"Final Episode Reward: {evaluate(Q)}")