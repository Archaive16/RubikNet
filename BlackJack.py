import gymnasium as gym
import numpy as np
from collections import defaultdict


env = gym.make("Blackjack-v1", sab=True)

# simple dunction to generate the episodes
def generate_episode(env, policy_fn):
    episode = []
    state, info = env.reset()
    while True:
        action = policy_fn(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        if terminated or truncated:
            break
        state = next_state
    return episode


# heart of the algorithm 
def mc_control_epsilon_greedy(env, episodes, epsilon):
    Q = defaultdict(lambda: np.zeros(2)) 
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    i=0
    # taking action using epsilon greedy strategy
    for i in range(episodes):
        def policy_fn(state):
            if np.random.rand() < epsilon:
                return np.random.choice([0, 1])
            else:
                return np.argmax(Q[state])
        i+=1

        episode = generate_episode(env, policy_fn)
        G = 0
        visited = set()

        for state, action, reward in reversed(episode):
            G = reward 
            # bellman or noraml mc equations
            if (state, action) not in visited:
                visited.add((state, action))
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

    
    final_policy = defaultdict(int)


    for state in Q:
        final_policy[state] = np.argmax(Q[state])
    
    return final_policy, Q


def evaluate_policy(env, policy_dict, episodes):
    def policy_fn(state):
        return policy_dict.get(state, np.random.choice([0, 1]))

    total_rewards = []
    i=0
    for i in range(episodes):
        state, info = env.reset()
        while True:
            action = policy_fn(state)
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                total_rewards.append(reward)
                break
        i+=1    

    no_of_games=len(total_rewards)

    num_wins=total_rewards.count(1)
    num_draw=total_rewards.count(0)
    num_loss=total_rewards.count(-1)

    win_rate=num_wins/no_of_games*100
    draw_rate=num_draw/no_of_games*100
    loss_rate=num_loss/no_of_games*100

    print(f"Win Rate:{win_rate}")
    print(f"Loss Rate:{loss_rate}") 
    print(f"Draw Rate:{draw_rate}")


print("Training policy with MC control...")
learned_policy, Q = mc_control_epsilon_greedy(env, episodes=10000, epsilon=0.1)

print("Evaluating learned policy...")
evaluate_policy(env, learned_policy, episodes=2000)
