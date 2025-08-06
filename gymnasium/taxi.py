import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Create environment
env = gym.make("Taxi-v3")

# Initialize Q-table
state_space = env.observation_space.n
action_space = env.action_space.n
q_table= np.zeros((500, 6))

# Hyperparameters
alpha = 0.1          
gamma = 0.99          
epsilon = 1.0         
epsilon_decay = 0.999 
epsilon_min = 0.05    
episodes = 10000      
win=0

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False
  
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if reward ==20:
            win+=1

        # Q-value update
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        q_table[state][action] += alpha * (td_target - q_table[state][action])
        state = next_state
        
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 500 == 0:
        print(f"Episode: {episode}, Success: {win}")

print("Training completed")


env_evaluate=gym.make("Taxi-v3", render_mode="rgb_array")
state,_=env_evaluate.reset()
done=False
total_reward=0
while not done:
    action=np.argmax(q_table[state])
    next_state, reward, terminated, truncated, _=env_evaluate.step(action)
    total_reward+=reward
    done=truncated or terminated
    state=next_state
    frame = env_evaluate.render()    # storing the frame
    plt.imshow(frame)
    plt.axis("off")
    plt.pause(1)
    
print(total_reward)
env_evaluate.close()






