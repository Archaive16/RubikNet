import numpy as np
import torch
from mcts import mcts
from cube import Cube
from state_capture import save_cube

cube = Cube()

if cube.is_solved():
    print("cube is already solved")

else:
    action_path, is_solved = mcts(cube.state, num_simulations=50, max_solution_depth=15)
    print(action_path)

    if is_solved and action_path:
        test_cube = Cube(state=cube.state.copy())
        moves_done = []  # store moves here

        for move_idx in action_path:
            if move_idx < len(test_cube.moves):
                move = test_cube.moves[move_idx]
                test_cube.move(move)
                moves_done.append(move)
                print(move)

        # Create/overwrite solution.txt and save moves
        with open("solution.txt", "w") as f:
            for move in moves_done:
                f.write(move + "\n")

        if test_cube.is_solved():
            print("Congratulations, Your solution is given above")
            print("Solution also saved in solution.txt")




