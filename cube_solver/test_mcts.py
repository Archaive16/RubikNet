import torch
import numpy as np
import random
import time
from cube import Cube
from adi import ADI

def get_cube_child_states(cube):
    children = []
    original_state = cube.state.copy()
    
    for move in cube.moves:
        child_cube = Cube(state=original_state.copy())
        child_cube.move(move)
        child_state = child_cube.state.copy()
        children.append(child_state)
    
    return children

def find_action_index_external(parent_state, child_state, cube_moves):
    try:
        if isinstance(parent_state, torch.Tensor):
            parent_decoded = decode_cube_state(parent_state)
        else:
            parent_decoded = parent_state
            
        if isinstance(child_state, torch.Tensor):
            child_decoded = decode_cube_state(child_state)
        else:
            child_decoded = child_state

        test_cube = Cube(state=parent_decoded.copy() if isinstance(parent_decoded, list) else list(parent_decoded))
        
        for i, move in enumerate(cube_moves):
            temp_cube = Cube(state=test_cube.state.copy())
            temp_cube.move(move)
            
            if temp_cube.state == child_decoded or np.array_equal(temp_cube.state, child_decoded):
                return i
        
    except Exception:
        pass
    
    return 0

color_map = {
    'w': [1, 0, 0, 0, 0, 0],
    'y': [0, 1, 0, 0, 0, 0],
    'b': [0, 0, 1, 0, 0, 0],
    'g': [0, 0, 0, 1, 0, 0],
    'r': [0, 0, 0, 0, 1, 0],
    'o': [0, 0, 0, 0, 0, 1]
}

def encode_cube_state(state_str):
    encoded = []
    for color in state_str:
        encoded.extend(color_map[color])
    return encoded

def decode_cube_state(encoded_state):
    color_list = ['w', 'y', 'b', 'g', 'r', 'o']
    decoded = []

    if isinstance(encoded_state, torch.Tensor):
        encoded_state = encoded_state.cpu().detach().numpy().flatten()

    for i in range(0, len(encoded_state), 6):
        color_vec = encoded_state[i:i+6]
        color_i = np.argmax(color_vec)
        decoded.append(color_list[color_i])
    return decoded


model = ADI()
model.load_state_dict(torch.load("deepcube_adi_model.pth", map_location='cpu'))
model.eval()


c = 1                     # exprolation constant
v = 50                    # virtual loss

def get_or_create_node(state, parent=None):
    if isinstance(state, torch.Tensor):
        state_key = tuple(state.cpu().numpy().flatten())
    else:
        state_key = str(state)
    
    if state_key in NODE._existing_nodes:
        return NODE._existing_nodes[state_key]
    else:
        new_node = NODE(state, parent)
        NODE._existing_nodes[state_key] = new_node
        return new_node

class NODE:
    _existing_nodes = {}
    
    def __init__(self, state, parent=None):
        self.state = state
        self.children_nodes = []
        self.N = 0
        self.W = 0
        self.L = 0
        self.P = 0
        self.U = 0
        self.Q = 0
        self.total = 0
        self.parent = parent
        
        try:
            with torch.no_grad():
                policy_logits, value = model(state)
            self.policy_logits = policy_logits.squeeze() if policy_logits.dim() > 1 else policy_logits
            self.value = value.item() if torch.is_tensor(value) else value
        except Exception:
            self.policy_logits = torch.ones(12)
            self.value = 0.0
        
        try:
            decoded_state = decode_cube_state(state)
            self.cube = Cube(state=decoded_state)
            self.is_solved = self.cube.is_solved()
        except Exception:
            self.cube = None
            self.is_solved = False
        
        if self.parent is not None and self.cube is not None:
            try:
                move_i = find_action_index_external(self.parent.state, self.state, self.cube.moves)
                if move_i < len(self.parent.policy_logits):
                    self.P = self.parent.policy_logits[move_i].item()
                else:
                    self.P = 0.1
            except Exception:
                self.P = 0.1
        
        self.update_values()
    
    def update_values(self):
        sum_N = sum(child.N for child in self.children_nodes)
        self.U = (c * self.P * (sum_N ** 0.5)) / (1 + self.N)
        self.Q = self.W - self.L
        self.total = self.Q + self.U
    
    def selection(self):
        if not self.children_nodes:
            return None
        
        for child in self.children_nodes:
            child.update_values()
        
        child_values = [child.total for child in self.children_nodes]
        best_child_idx = np.argmax(child_values)
        
        self.N += 1
        self.L += v
        
        return best_child_idx
    
    def backpropagate(self, value):
        self.W = max(self.W, value)
        # self.L = max(0, self.L)
        
        if self.parent is not None:
            self.parent.backpropagate(value)
    
    def expand(self):
        if self.children_nodes or self.cube is None:
            return
        
        try:
            children = get_cube_child_states(self.cube)
            
            for i, child_state in enumerate(children):
                if not isinstance(child_state, list) or len(child_state) != 54:
                    continue
                    
                if not all(isinstance(color, str) for color in child_state):
                    continue
                
                child_str = ''.join(child_state)
                
                try:
                    child_encoded = encode_cube_state(child_str)
                except Exception:
                    continue
                
                child_tensor = torch.FloatTensor(child_encoded).unsqueeze(0)
                child_node = get_or_create_node(child_tensor, self)
                self.children_nodes.append(child_node)
            
            self.update_values()
            
        except Exception:
            pass

def mcts(state, num_simulations, max_solution_depth):
    NODE._existing_nodes = {}
    
    try:
        if isinstance(state, list):
            state_encoded = encode_cube_state(''.join(state))
        else:
            state_encoded = encode_cube_state(state)
        
        state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0)
        
    except Exception:
        return [], False
    
    try:
        root = NODE(state_tensor)
    except Exception:
        return [], False
    
    if root.is_solved:
        return [], True
    
  
    # queue = [(root, [])]
    # visited = set()
    
    # while queue and len(queue[0][1]) <= max_solution_depth:
    #     current_node, path = queue.pop(0)
        
    #     state_key = tuple(current_node.state.cpu().numpy().flatten())
    #     if state_key in visited:
    #         continue
    #     visited.add(state_key)
        
    #     if not current_node.children_nodes:
    #         current_node.expand()
        
    #     for i, child in enumerate(current_node.children_nodes):
    #         new_path = path + [i]
            
    #         if child.is_solved:
    #             return new_path, True
            
    #         if len(new_path) < max_solution_depth:
    #             queue.append((child, new_path))
    
   
    solution_found = False
    solution_node = None
    
    for simulation in range(num_simulations):
        current = root
        path = [current]
        depth = 0
        max_depth = 150
        do_again = True
        while do_again:
            if current.children_nodes and not current.is_solved and depth < max_depth:
                action_idx = current.selection()
                if action_idx is None:
                    break
                value = current.value
                current = current.children_nodes[action_idx]
                path.append(current)
                for node in reversed(path):
                    node.backpropagate(value)
                depth += 1
            
                if current.is_solved:
                    solution_found = True
                    solution_node = current
                    break
        
            if solution_found:
                break
            
            if not current.is_solved and not current.children_nodes:
                current.expand()
            
                for i, child in enumerate(current.children_nodes):
                    if child.is_solved:
                        solution_found = True
                        solution_node = child
                        break
            
                if not solution_found and current.children_nodes:
                    child_values = []
                    for child in current.children_nodes:
                        child.update_values()
                        child_values.append(child.value + child.U)
                
                    if child_values:
                        best_child_idx = np.argmax(child_values)
                        value = current.value
                        current = current.children_nodes[best_child_idx]
                        path.append(current)
                        for node in reversed(path):
                            node.backpropagate(value)
                        depth += 1
        
            if solution_found:
                break
            elif depth > max_depth:
                do_again = False
        
        # for node in reversed(path):
        #     node.backpropagate(value)
    
    if solution_found and solution_node:
        solution_path = []
        temp_node = solution_node
        while temp_node.parent is not None:
            parent = temp_node.parent
            for i, child in enumerate(parent.children_nodes):
                if child == temp_node:
                    solution_path.append(i)
                    break
            temp_node = parent
        solution_path = list(reversed(solution_path))
        return solution_path, True
    
    return [], False