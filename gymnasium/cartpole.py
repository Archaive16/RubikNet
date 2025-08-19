import gymnasium as gym 
import numpy as np
import math 

env= gym.make("CartPole-v1")
q_table=np.zeros((10,10,10,10,2))

lower_bounds = [-4.8, -3.0, -0.418, -math.radians(50)]
upper_bounds = [4.8, 3.0, 0.418, math.radians(50)]

def discritsize(obs):
    cart_pos, cart_vel, pole_pos, pol_vel=obs
    cart_pos_scaled=(cart_pos+4.8)/9.6
    cart_vel_scaled=(cart_vel+3)/6
    pole_pos_scales=(pole_pos+0.418)/0.836
    pol_vel_scaled=(pol_vel+math.radians(50))/(2*math.radians(50))

    cart_pos_want = int(np.clip(cart_pos_scaled * 10, 0, 9))
    cart_vel_want = int(np.clip(cart_vel_scaled * 10, 0, 9))
    pole_pos_want = int(np.clip(pole_pos_scales * 10, 0, 9))
    pole_vel_want = int(np.clip(pol_vel_scaled * 10, 0, 9))

    return cart_pos_want, cart_vel_want, pole_pos_want, pole_vel_want

def calculate_reward(action, current_obs, next_obs):
    reward = 0

    # 1. Reward if pole is more vertical
    if abs(next_obs[2]) < 0.05:
        reward += 5
    elif abs(next_obs[2]) < 0.1:
        reward += 1
    else:
        reward -= 2

    # 2. Reward if cart is near center
    if abs(next_obs[0]) < 0.25:
        reward += 2
    elif abs(next_obs[0]) < 0.5:
        reward += 0
    else:
        reward -= 1

    # 3. Small penalty for high angular velocity (pole speed)
    if abs(next_obs[3]) > 0.75:
        reward -= 2

    return reward

    
    

alpha = 0.1             
gamma = 0.99            
epsilon = 1.0           
min_epsilon = 0.05
epsilon_decay = 0.98
episodes = 10000
max_steps = 500

for episode in range(episodes):
    obs, _ = env.reset()
    state = discritsize(obs)
    total_reward = 0

    for step in range(max_steps):
        # Choose action: epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, _, done, truncated, _ = env.step(action)
        reward=calculate_reward(action,obs,next_obs)
        next_state = discritsize(next_obs)

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

env=gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()
state = discritsize(obs)
done=False

i=1
while not done:
    env.render()
    action = np.argmax(q_table[state])
    obs, reward, done, truncated, _ = env.step(action)
    state = discritsize(obs)
    
    

env.close()
