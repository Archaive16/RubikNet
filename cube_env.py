import numpy as np
import random

class Cube:
    def __init__(self):
        self.reset_cube()
        self.move_history = []

    def reset_cube(self):
        self.state = ["w"]*9 + ["y"]*9 + ["b"]*9 + ["g"]*9 + ["r"]*9 + ["o"]*9

    def is_solved(self):
        for i in range(0, 54, 9):
            if len(set(self.state[i:i+9])) != 1:
                return False
        return True

    def scramble(self, n):
        moves = ["U", "D", "F", "B", "R", "L", "U'", "D'", "F'", "B'", "R'", "L'"]
        sequence = [random.choice(moves) for _ in range(n)]
        for move in sequence:
            self.move(move)
        print("Scramble moves:", ' '.join(sequence))

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



cube=Cube()
cube.print_cube()
print(cube.is_solved())
cube.scramble(5)
cube.reset_cube()
cube.print_cube()
print(cube.is_solved())
cube.scramble(5)
print(cube.move_history)