import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v1', sab=True) #creating our env 
#1) epsilon greedy policy generator 10% explore and 90%exploit but done in random order to not avoid later good rewards or possiblites 
def epsilon_greedy(Q,nA,epsilon):
    # This function returns another function (policyf) that picks actions
    # given the current Q-table, number of actions, and epsilon.

    def policyf (state):
        # pick random btw 1-0
        rand_val = np.random.rand()
        if rand_val < epsilon:
            action = np.random.choice(nA)
        else:
            # Exploit: pick the action with highest estimated value
            # Look up Q-values for the state
            q_values = Q[state]
            # Find the (action) of the maximum Q
            max_index = 0
            max_value = q_values[0]
            for i in range(1, nA):
                if q_values[i] > max_value:
                    max_value = q_values[i]
                    max_index = i
            action = max_index
        return action
    return policyf

#2 to generate episodes 
def genrate_ep(env , policy):
    observation, info = env.reset()
    state = observation  # state is a tuple (player_sum, dealer_card, usable_ace)
    episode = []         # to store (state, action, reward)
    done = False
    
    while not done:
        # action using policy
        action = policy(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # episode ends
        done = terminated or truncated
        # store 
        episode.append((state, action, reward))
        # next state
        state = next_obs

    return episode

#3) here we go with Monte Carlo control: alternates MC prediction of Q under current ε‑greedy pi and policy improvement (ε‑greedy w.r.t. Q)
# +with backview which satisfes conditions for mcc 
# get G >>update Q values >> evaluate policy 
def mccontrol(env , num_episodes, gamma=1.0 , epsilon=0.1):
     # initialize Q-table== maps each state to an array of action-values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # To accumulate returns and counts for averaging
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # initial policy
    policy = epsilon_greedy(Q, env.action_space.n, epsilon)
    # For plotting
    episode_returns = []
#looping over episodes 
    for episode_idx in range(1, num_episodes + 1)
        episode = genrate_ep(env , policy)
        #  total reward for this episode
        total_reward = 0
        for (s, a, r) in episode:
            total_reward += r
        episode_returns.append(total_reward)

        G = 0
        visited = set()  # track first visits
        # episode in reverse order backward view 
        for step in reversed(episode):
            s, a, r = step
            G = gamma * G + r
            # First-visit check
            if (s, a) not in visited:
                # Update sums and counts
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1
               
              # Update Q-value as average
                #Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]
               
                alpha = 0.1
                Q[s][a] += alpha * (G - Q[s][a])  #better than taking average for q values 
                visited.add((s, a))

        # Update policy to new Q
        policy = epsilon_greedy(Q, env.action_space.n, epsilon)
    
    #final policy 
    final={}
    for state,action_values in Q.items():
        bestaction=0
        bestvalue=action_values[0]
        for i in range(1, len(action_values)):
          if action_values[i]> bestvalue:
             bestvalue=action_values[i]
             bestaction=i
        final[state]=bestaction
    return Q, final, episode_returns  

num_episodes = 500_000
Q, policy, episode_returns = mccontrol(env, num_episodes)

#  Evaluate learned policy
wins = 0
num_games = 10_000
for _ in range(num_games):
    obs, _ = env.reset()
    state = obs
    done = False
    while not done:
        # If state not seen >>> standing (action 0)
        if state in policy:
            action = policy[state]
        else:
            action = 0
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    if reward > 0:
        wins += 1
print("Win rate: " + str(wins / num_games))



#  Plot training progress with zoomed-in view(taken help here for plotting)
plt.figure(figsize=(10, 5)) 
plt.plot(episode_returns, alpha=0.2, color='blue')

window = 500 
moving_avg = np.convolve(episode_returns, np.ones(window)/window, mode='valid')

plt.plot(range(window-1, len(episode_returns)), moving_avg, 
         linewidth=2, color='darkorange',
         label=f'{window}-episode moving average')

reward_range = (-1, 1)
plt.ylim(reward_range)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Episode Returns Over Time (Detailed View)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout() 
plt.show()





