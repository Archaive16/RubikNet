# Autodidactic Iteration

*Learning backwards from the goal*

## The Question

While deep RL has conquered Chess, Go, and countless Atari games, these victories came with a crucial advantage: frequent feedback. Every captured piece in Chess, every territory gained in Go provides a learning signal. But what happens when an AI faces problems where success is measured by a single, distant goal?

The Rubik's Cube worsens this probability. With approximately 4.3 × 10¹⁹ possible configurations and only one solved state, the probability of stumbling upon success randomly is:

```
P(random_success) = 1 / (4.3 × 10¹⁹) ≈ 2.3 × 10⁻²⁰
```
At a billion attempts per second, you'd expect success once every 1.4 billion years. So how is an agent going to stumble upon the single reward generating state and actually find something meaningful?

## The Idea

If the reward is practically impossible to get, why don't we begin with the reward state itself? Therefore, some states would be close enough to the solution to stumble upon it and creating a something of value for the other states to follow.

## The Bug: Sparse Rewards

**Exploration is pointless.** In massive state spaces, random exploration is meaningless. Without intermediate signals, agents wander aimlessly in a never-ending environment.

**Credit assignment vanishes.** When rewards appear an unknown amount of steps away from crucial decisions, which actions actually mattered? This uncertainty makes learning nearly impossible.

**Convergence seems like chance.** Algorithms either diverge completely or converge to degenerate policies that avoid exploration altogether.

### Real-World Concerns

This isn't just about basic game-like settings. Sparse reward environments dominate the problems that matter most:

- **Scientific Discovery**: Molecular design where only specific arrangements show biological activity
- **Engineering**: Circuit layouts where optimal configurations exist among billions of possibilities  
- **Logistics**: Resource allocation where perfect efficiency emerges from countless variable combinations

## The Root: Self-Supervised Learning

Before diving into our solution, it's worth understanding why self-supervised learning has emerged as such a powerfool tool applicable to almost any problem we come across. It tackles one of the biggest challenges, we simply don't have enough labeled data for most problems. Instead of needing humans to tag everything, these systems learn to teach themselves from raw, unlabeled data.

The trick is giving models clever self-created puzzles. Show them an image with parts covered up and ask what's missing. Give them half a sentence and have them complete it. Through these exercises, they start recognizing deep patterns in the data without any human guidance.

This approach takes the spotlight in complex scenarios like the Rubik's Cube, which has 43 quintillion possible arrangements. Rather than trying to memorize solutions for each configuration, the system learns to generalize. This lets it solve scrambles it's never seen before.

## The Core

Autodidactic Iteration (ADI) represents a fundamental shift in how we approach sparse reward learning. The breakthrough insight is astonishingly naive, instead of searching forward from random states toward an unknown goal, why not start from the goal and work backward?

This reframes the entire setting. It helps the network see data in a way that would have been impossible before.

**Traditional Setting:** Chance becomes the only possible hope.
```
Random State → Action → No feedback → ... → Goal State
```

**ADI:**
```
Goal State ─┬─→ Action → State 1 → Action → State 2 → ...
            ├─→ Action → State → ...
            ├─→ ...
            ├─→ ...
            .
            .
```
This process is repeated many times to generate data.

```
                           ╭─→ ..
                           .
                           .
                           ├─→ ..
      State 1 ─────────────┤
                           ├─→ Goal State
                           .
                           .
                           ╰─→ ..
```
Some of the data points are close enough to find the solution by chance.

### The Math 

ADI trains a neural network f_θ(s) that outputs both:
- **Value v(s)**: How close is this state to the solution?
- **Policy p(s)**: What's the best move to make?

The training process follows a simple loop:

1. **Generate Training Data**: Start from solved state, apply k random scrambles (as shown above)
2. **Compute Targets**: For each state, evaluate all possible next moves using current network
3. **Bootstrap Learning**: Use network's own predictions as training targets

The target computation follows the Bellman equation:
```
Value Target: y^v = max_a [R(child_a) + v(child_a)]
Policy Target: y^p = argmax_a [R(child_a) + v(child_a)]
```

### Why It Works: The Self-Supervision Connection

ADI embodies the essence of self-supervised learning in the RL domain. Self-supervised learning is an evolving machine learning technique essentially to solve the challenges posed by the over-dependence of labeled data. By generating its own training examples from backward trajectories, ADI eliminates the need for external supervision.

## Our Implementation Journey

All of these concepts played a crucial role in implementing ADI to solve a Rubik's Cube. A challenge, seeming genuinely impossible, turned out to be so rudimentary.

### Network Architecture

Our neural network design:

```python
class ADI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(324, 4096)    # First hidden layer
        self.fc2 = nn.Linear(4096, 2048)   # Second hidden layer
        
        # Policy Head
        self.fc3_1 = nn.Linear(2048, 512)
        self.fc_policy_out = nn.Linear(512, 12)
        
        # Value Head
        self.fc3_2 = nn.Linear(2048, 512)
        self.fc_value_out = nn.Linear(512, 1)
```

```python
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        # Policy Path
        policy_logits = F.elu(self.fc3_1(x))
        policy_logits = self.fc_policy_out(policy_logits)

        # Value Path
        value = F.elu(self.fc3_2(x))
        value = torch.tanh(self.fc_value_out(value))

        return policy_logits, value
```

**Why This Architecture?**
- **324 inputs**: Cube state is one hot encoded into an array of 324 digits {0, 1}
- **Shared Main Network**: Different neural network for each might create discrepancies
- **Dual heads**: Specialized pathways allow specific optimizability
- **ELU activations**: Smooth gradients and no dead neuron problems
- **Tanh output for value**: Bounds predictions to [-1, 1] range for stability

### Initialization

```python
    def apply_glorot_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
```
Xavier initialization with 0.5 gain prevents vanishing/exploding gradients in the deep network and maintains stable activation distributions through ELU layers.

### Target Generation

```python
def targets(states, pred_vals, device='cpu'):
    action_vals = []
    
    with torch.no_grad():
        for i, state in enumerate(states):
            cube = Cube(state)
            reward = cube.get_reward()
            pred_val = torch.clamp(pred_vals[i].squeeze(), -1, 1).item()
            
            action_val = reward + pred_val
            action_vals.append(action_val)
    
    # Select best action and value
    max_i = np.argmax(action_vals)
    target_val = np.tanh(action_vals[max_i])
    target_policy_i = torch.tensor(max_i, dtype=torch.long, device=device)
    
    return target_policy_i, target_val
```

This function basically uses the output (predictions) of the Child states to compute targets for the Base state (it originated from). This is the fundamental game-changing concept of ADI, doing that one-step look-ahead and figuring the value and policy of the given state by comparing all the states formed by it.

### Training Strategy

```python
# Multi-component optimizer with different learning rates
optimizer = optim.AdamW([
    {'params': policy_params, 'lr': 1e-5, 'weight_decay': 1e-4},
    {'params': value_params, 'lr': 5e-5, 'weight_decay': 1e-4},
    {'params': shared_params, 'lr': 1e-5, 'weight_decay': 5e-5}
])
```
- Different parameter groups allow fine-tuned learning, policy head gets 5x lower LR since it was observed that policy was overfitting whithin a few iterations.
- Shared LR (1e-5) is conservative to maintain stable updates too.
- Weight decay values are related to how specailized those layers are, shared params get lower decay (5e-5) since they're critical for both tasks, while heads get higher decay (1e-4) to prevent overfitting and penalize more.

### Weighted loss based on distance from solution
```
W = 1 / max(depth, 1.0)
weighted_policy_loss = W * loss_policy
weighted_value_loss = W * loss_val
```
We value the states closer to the Goal State (Solution) more rather than far-fetched ones.

**Key Properties:**

**Fresh Data Every Epoch**: We generate new scrambled states for each training epoch. This prevents overfitting and ensures the network continuously adapts to new challenges.

**Distance-Based Weighting**: States closer to the solution receive higher training weights. This creates a natural hierarchy and labels the importance of each state in the actual learning process.

**Difference in Learning Rates and Weight Decay**: Different network components require different approaches. Value is learnt more rigorously compared to the policy. Shared layers contribute less and therefore can do with less penalization.

### The Training

Finally after,
- **100 epochs** of training
- **1000 scrambled runs per epoch** 
- **20-move scramble depth** - resulting in 20,000 state configurations every epoch
- **Multiple iterations through all**
- **___ hours of continuous GPU training**                    **have to add in the actual hours took**

Each epoch processes 240,000 child states (20000 cubes × 12 moves each), generating over millions of value and policy predictions. The network witnesses billions of training examples throughout the entire process.

### Major Debugs

**The "Seems like we got it" Problem**: Getting a tremendously low loss after just a few epochs with a naive prototype version of the present code, turns out the model did not learn a single thing. It was just overfitting and never really learnt anything useful. Solution: change the main optimizer function and the right allocation of learning rates and weight decay.

**Memory Management**: Processing thousands of cube states simultaneously pushed GPU memory limits. Solution: careful batch management and periodic cache clearing.

**Gradient Instability**: When it wasn't overfitting, it was exploding. Solution: gradient clipping at 0.5 norm and more careful learning rate scheduling.

```python
# Gradient clipping for stability
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

# Periodic memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```
*And thus, our ADI implementation worked wonders and combined with MCTS could solve the Rubik's Cube with a success rate of **100%** for any given scrambled state*

## The Broader Implications

ADI's backward-learning breakthrough  can be applied to numerous fields where sparse rewards have traditionally made learning impossible:

**Drug Discovery**: Start with known effective compounds and work backward through molecular modifications to understand structure-activity relationships.

**Protein Folding**: Begin with stable protein structures and learn which conformational changes lead to instability, building understanding of folding landscapes.

**Circuit Optimization**: Take high-performing chip layouts and introduce perturbations to learn what makes designs effective.

**Logistics Planning**: Start from optimal resource allocations and work backward through disruptions to build robust contingency strategies.

## Current Limitations and Future Horizons

### What We Learned About Constraints

**Computational Intensity**: ADI demands significant computational resources. Our implementation itself required millions of training samples. Scaling to larger problems remains challenging.

**Reversibility**: ADI works best when actions are easily reversible, allowing clean backward generation. Not all real-world problems necessarily have this property.

**Hyperparameter Sensitivity**: Success depends critically on learning rates, batch sizes, scramble depths, and loss weighting. These parameters require careful tuning for each problem domain.

## Conclusion

