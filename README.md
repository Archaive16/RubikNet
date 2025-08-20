# RubikNet 🧩

RubikNet is a project exploring how AI and robotics can solve the Rubik’s Cube using **Reinforcement Learning**, **Deep Learning**, and **Search-based Solvers**.  
This repository is organized into three main components:

---

## 📂 Reinforcement Learning
This folder contains experiments and implementations of **RL agents** trained to solve the Rubik’s Cube environment.  
- Uses Gymnasium-based cube simulations.  
- Implements algorithms like Monte Carlo Tree Search (MCTS) and policy/value networks.  
- Focus on **learning from self-play** and **reward engineering**.  

---

## 📂 Deep Learning
This folder explores **neural network models** for cube state prediction and policy approximation.  
- Includes supervised learning experiments (state → move prediction).  
- Explores architectures like CNNs/MLPs for cube representation.  
- Supports integration with the RL agent for faster convergence.  

---

## 📂 Cube Solver
This folder contains the **solver logic** for the Rubik’s Cube.  
- Includes **BFS/MCTS-based solvers** and classical search approaches.  
- Can be used standalone to find solutions to scrambled states.  
- Forms the backbone for testing learned models against logical solvers.  

---

## 🚀 Usage
We recommend using [uv](https://docs.astral.sh/uv/) for environment management.  
To set up the project:  

```bash
# Install uv (if not already installed)
pip install uv

# Sync dependencies
uv sync
```

## 🔗 Connect with us


#### Arhan Chavare  
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Archaive16)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arhan-chavare-5a23a8334/)  

### Rigvedi Borchate  
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rigvedi-3301)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rigvedi-borchate-50b210330/)  




  
