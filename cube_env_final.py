import numpy as np
import random
import csv

class Cube:
    def __init__(self, state=None):
        self.reset_cube()
        self.move_history = []
        self.scramble_states = [] 
        self.moves=["U", "D", "F", "B", "R", "L", "U'", "D'", "F'", "B'", "R'", "L'"]
        if state is not None:
            self.state = state.copy()

    def reset_cube(self):
        self.state = ["w"]*9 + ["y"]*9 + ["b"]*9 + ["g"]*9 + ["r"]*9 + ["o"]*9
        self.history = []
    
    def is_solved(self):
        if self.state is None:
            return False
        for i in range(0, 54, 9):
            if len(set(self.state[i:i+9])) != 1:
                return False
        return True
    
    def scramble(self, n):
        moves = self.moves
        self.scramble_states = [] 
        self.move_history = []
        for turn in range(n):
            move = random.choice(moves)
            self.move(move)
            self.scramble_states.append({
                "state": self.state.copy(),
                "moves": self.move_history.copy(),
                "turns": turn + 1
            })
        print("Scramble moves:", ' '.join(self.move_history))


    def move(self, move_name):
        self.move_history.append(move_name)
        self.state = self.switch(move_name, self.state)

    def print_cube(self):
        print("F:", self.state[0:9])
        print("B:", self.state[9:18])
        print("T:", self.state[18:27])
        print("D:", self.state[27:36])
        print("R:", self.state[36:45])
        print("L:", self.state[45:54])


    def get_reward(self):
        return 1 if self.is_solved() else -1
    
    def switch(self, move, cube):
        cube = cube.copy()

        def rotate_face(c, a,b,c_,d,e,f,g,h):
            c[a],c[b],c[c_],c[d],c[e],c[f],c[g],c[h] = c[c_],c[e],c[h],c[b],c[g],c[a],c[d],c[f]

        def rotate_face_anti(c, a,b,c_,d,e,f,g,h):
            c[a],c[b],c[c_],c[d],c[e],c[f],c[g],c[h] = c[f],c[d],c[a],c[g],c[b],c[h],c[e],c[c_]

        if move == "F":
            rotate_face_anti(cube, 0,1,2,3,5,6,7,8)
            cube[24],cube[25],cube[26],cube[36],cube[39],cube[42],cube[33],cube[34],cube[35],cube[53],cube[50],cube[47] = \
            cube[53],cube[50],cube[47],cube[24],cube[25],cube[26],cube[36],cube[39],cube[42],cube[33],cube[34],cube[35]

        elif move == "F'":
            rotate_face(cube, 0,1,2,3,5,6,7,8)
            cube[24],cube[25],cube[26],cube[36],cube[39],cube[42],cube[33],cube[34],cube[35],cube[53],cube[50],cube[47] = \
            cube[36],cube[39],cube[42],cube[33],cube[34],cube[35],cube[53],cube[50],cube[47],cube[24],cube[25],cube[26]

        elif move == "B":
            rotate_face_anti(cube, 9,10,11,12,14,15,16,17)
            cube[18],cube[19],cube[20],cube[45],cube[48],cube[51],cube[29],cube[28],cube[27],cube[44],cube[41],cube[38] = \
            cube[44],cube[41],cube[38],cube[18],cube[19],cube[20],cube[45],cube[48],cube[51],cube[29],cube[28],cube[27]

        elif move == "B'":
            rotate_face(cube, 9,10,11,12,14,15,16,17)
            cube[18],cube[19],cube[20],cube[45],cube[48],cube[51],cube[29],cube[28],cube[27],cube[44],cube[41],cube[38] = \
            cube[45],cube[48],cube[51],cube[29],cube[28],cube[27],cube[44],cube[41],cube[38],cube[18],cube[19],cube[20]

        elif move == "U":
            rotate_face_anti(cube, 18,19,20,21,23,24,25,26)
            cube[0],cube[1],cube[2],cube[36],cube[37],cube[38],cube[9],cube[10],cube[11],cube[45],cube[46],cube[47] = \
            cube[36],cube[37],cube[38],cube[9],cube[10],cube[11],cube[45],cube[46],cube[47],cube[0],cube[1],cube[2]

        elif move == "U'":
            rotate_face(cube, 18,19,20,21,23,24,25,26)
            cube[0],cube[1],cube[2],cube[36],cube[37],cube[38],cube[9],cube[10],cube[11],cube[45],cube[46],cube[47] = \
            cube[45],cube[46],cube[47],cube[0],cube[1],cube[2],cube[36],cube[37],cube[38],cube[9],cube[10],cube[11]

        elif move == "D":
            rotate_face_anti(cube, 27,28,29,30,32,33,34,35)
            cube[6],cube[7],cube[8],cube[51],cube[52],cube[53],cube[15],cube[16],cube[17],cube[42],cube[43],cube[44] = \
            cube[42],cube[43],cube[44],cube[6],cube[7],cube[8],cube[51],cube[52],cube[53],cube[15],cube[16],cube[17]

        elif move == "D'":
            rotate_face(cube, 27,28,29,30,32,33,34,35)
            cube[6],cube[7],cube[8],cube[51],cube[52],cube[53],cube[15],cube[16],cube[17],cube[42],cube[43],cube[44] = \
            cube[51],cube[52],cube[53],cube[15],cube[16],cube[17],cube[42],cube[43],cube[44],cube[6],cube[7],cube[8]

        elif move == "R":
            rotate_face_anti(cube, 36,37,38,39,41,42,43,44)
            cube[2],cube[5],cube[8],cube[26],cube[23],cube[20],cube[11],cube[14],cube[17],cube[29],cube[32],cube[35] = \
            cube[29],cube[32],cube[35],cube[2],cube[5],cube[8],cube[26],cube[23],cube[20],cube[11],cube[14],cube[17]

        elif move == "R'":
            rotate_face(cube, 36,37,38,39,41,42,43,44)
            cube[2],cube[5],cube[8],cube[26],cube[23],cube[20],cube[11],cube[14],cube[17],cube[29],cube[32],cube[35] = \
            cube[26],cube[23],cube[20],cube[11],cube[14],cube[17],cube[29],cube[32],cube[35],cube[2],cube[5],cube[8]

        elif move == "L":
            rotate_face_anti(cube, 45,46,47,48,50,51,52,53)
            cube[0],cube[3],cube[6],cube[18],cube[21],cube[24],cube[9],cube[12],cube[15],cube[33],cube[30],cube[27] = \
            cube[18],cube[21],cube[24],cube[9],cube[12],cube[15],cube[33],cube[30],cube[27],cube[0],cube[3],cube[6]

        elif move == "L'":
            rotate_face(cube, 45,46,47,48,50,51,52,53)
            cube[0],cube[3],cube[6],cube[18],cube[21],cube[24],cube[9],cube[12],cube[15],cube[33],cube[30],cube[27] = \
            cube[33],cube[30],cube[27],cube[0],cube[3],cube[6],cube[18],cube[21],cube[24],cube[9],cube[12],cube[15]

        else:
            print(f"Invalid move: {move}")

        return cube
    
    def get_child_states_at_all_steps(self):
        all_children = []
        for scramble_state in self.scramble_states:
            base_state = scramble_state["state"]
            children = []
            for move in self.moves:
                new_cube = Cube(state=base_state.copy())
                new_cube.move(move)
                children.append(new_cube)
            all_children.append(children)
        return all_children




def generate_dataset(k, L, filename="cube_dataset.csv"):
    cube = Cube()
    moves = cube.moves

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        
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


generate_dataset(k=3,L=1) 

