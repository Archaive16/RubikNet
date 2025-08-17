import csv
from cube import Cube

def generate_dataset(k, L, filename="cube_dataset.csv"):
    # Generate a dataset of cube states and moves.
    
    cube = Cube()
    moves = cube.moves

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow([
            "scramble_run", "scramble_step", "base_move",
            "child_move", "base_state", "child_state"
        ])

        for scramble_run in range(1, L+1):
            cube.reset_cube()
            cube.scramble(k)

            all_children = cube.get_child_states_at_all_steps()

            for step_num, children in enumerate(all_children, start=1):
                base_state = cube.scramble_states[step_num-1]["state"]
                base_move = cube.scramble_states[step_num-1]["moves"][-1]

                for child_move, child_cube in zip(moves, children):
                    writer.writerow([
                        scramble_run, step_num, base_move,
                        child_move,
                        ''.join(base_state),
                        ''.join(child_cube.state)
                    ])

    print(f"Dataset saved to {filename}")