import numpy as np
import gymnasium as gym
from collections import defaultdict

# 1. monte carlo fn --> generate episode --> pick action 
# 2. calculate win rate

env = gym.make("Blackjack-v1", sab=True)

def monte_carlo(n_episodes, epsilon, alpha, gamma, Q):
    for _ in range(n_episodes):
        episode = generate_ep(Q, epsilon)    # output = tuple(states, actions, rewards)
        states, actions, rewards = zip(*episode)
        epsilon *= 0.99995    # decay epsilon
        n = 0
        for state in states:
            discounted_return = 0
            k = np.arange(len(rewards[n:]))
            for j in range(len(rewards[n:])):
                discounted_return += rewards[n + j] * (gamma**k[j])    # discounted return = R(t) + gamma * R(t+1) + ...
            Q[state][actions[n]] += alpha * (discounted_return - Q[state][actions[n]])    # bellman equation
            n += 1
    return Q

def generate_ep(Q, epsilon):
    state, _ = env.reset()
    episode = []

    completed = 0
    while not completed:
        action = generate_action(Q, epsilon, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        completed = terminated or truncated
        episode.append((state, action, reward))
        state = next_state
    
    return episode

def generate_action(Q, epsilon, state):
    action_space = [0, 1]
    probabilities = [1 - epsilon / 2, epsilon / 2] if np.argmax(Q[state]) == 0 else [epsilon / 2, 1 - epsilon / 2]
    action = np.random.choice(action_space, p = probabilities)
    return action

def get_win_rate(Q, n_episodes):
    # win rate = num(total reward > 0) / total episodes
    episodes_won = 0
    for _ in range(n_episodes):
        total_reward = 0
        completed = 0
        while not completed:
            state, _ = env.reset()
            action = np.argmax(Q[state]) if state in Q else np.random.choice([0, 1])
            next_state, reward, terminated, truncated, _ = env.step(action)
            completed = terminated or truncated
            total_reward += reward
            state = next_state
        if total_reward > 0:
            episodes_won += 1
    return (episodes_won / n_episodes) * 100

Q = defaultdict(lambda: np.zeros(2))
print("Training...")
Q = monte_carlo(100000, 1.0, 0.4, 0.9999, Q)

print("Calculating...")
print(f"Win Rate: {get_win_rate(Q, 10000)}")