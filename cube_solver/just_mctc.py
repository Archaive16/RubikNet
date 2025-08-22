from cube import Cube 
from adi import ADI
import time
import torch
from mcts import mcts, NODE , get_cube_child_states, find_action_index_external, encode_cube_state, decode_cube_state
solved =0
num_test=10

model = ADI()
model.load_state_dict(torch.load("deepcube_adi_model.pth", map_location='cpu'))
model.eval()


for i in range(num_test):
    cube=Cube()
    cube.scramble(5)
    print(cube.move_history)

    action_path, is_solved = mcts(cube.state, num_simulations=500, max_solution_depth=15)
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
        

        if test_cube.is_solved():
            print("Congratulations, Your solution is given above")
            
