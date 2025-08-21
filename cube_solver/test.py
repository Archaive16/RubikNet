from adi import test_model
from cube import Cube
from mcts import mcts, NODE


solved = 0

for i in range(1):
    cube = Cube()
    cube.reset_cube()
    cube.scramble(6)
    print(cube.move_history)
    
    if cube.is_solved():
        continue
    
    action_path, is_solved = mcts(cube.state, num_simulations=50000, max_solution_depth=105)
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


print(solved)


