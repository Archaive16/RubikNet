import torch
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Assuming the Cube class and ADI model are defined as in the previous code

# Load the trained model
model = ADI()
model.load_state_dict(torch.load('deepcube_adi_model.pth'))
model.to(device)
model.eval()

# Function to generate scrambled cubes
def generate_scrambled_cubes(num_cubes, scramble_depth):
    cubes = []
    for _ in range(num_cubes):
        cube = Cube()
        cube.scramble(scramble_depth)
        cubes.append(cube)
    return cubes

# Function to predict the next move using the model
def predict_next_move(model, cube_state):
    encoded_state = encode_cube_state(cube_state)
    encoded_state = torch.FloatTensor(encoded_state).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _ = model(encoded_state)
        policy_probs = F.softmax(policy_logits, dim=1)
        next_move_index = torch.argmax(policy_probs, dim=1).item()
    return cube.moves[next_move_index]

# Function to solve the cube using the model
def solve_cube(model, cube, max_moves=1000):
    move_history = []
    for _ in range(max_moves):
        if cube.is_solved():
            break
        next_move = predict_next_move(model, cube.state)
        cube.move(next_move)
        move_history.append(next_move)
    return move_history, cube.is_solved()

# Generate 100 scrambled cubes with a scramble distance of 10
num_cubes = 100
scramble_depth = 10
scrambled_cubes = generate_scrambled_cubes(num_cubes, scramble_depth)

# Solve each scrambled cube using the model
solved_count = 0
for i, cube in enumerate(scrambled_cubes):
    print(f"Solving cube {i+1}/{num_cubes}...")
    move_history, is_solved = solve_cube(model, cube)
    if is_solved:
        solved_count += 1
        print(f"Cube {i+1} solved in {len(move_history)} moves.")
    else:
        print(f"Cube {i+1} not solved within {max_moves} moves.")

print(f"\nTotal cubes solved: {solved_count}/{num_cubes}")