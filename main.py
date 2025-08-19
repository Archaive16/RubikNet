import numpy as np
import torch
from mcts import mcts
from cube import Cube


solved = 0

for i in range(20):
    cube = Cube()
    cube.reset_cube()
    cube.scramble(4)
    print(cube.move_history)
    
    if cube.is_solved():
        continue
    
    action_path, is_solved = mcts(cube.state, num_simulations=50, max_solution_depth=15)
    print(action_path)
    
    if is_solved and action_path:
        test_cube = Cube(state=cube.state.copy())
        
        for move_idx in action_path:
            if move_idx < len(test_cube.moves):
                test_cube.move(test_cube.moves[move_idx])
                print(test_cube.moves[move_idx])
        
        if test_cube.is_solved():
            solved += 1
    elif is_solved:
        solved += 1

print(f"Solved {solved}/20 cubes ")