import numpy as np 
import torch 
from cube import Cube
from adi import ADI

model=ADI()
model.load_state_dict(torch.load('deepcube_adi_model.pth'), map_location='cpu')
model.eval()

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

def _to_state_list(state):
    
    if isinstance(state, list) and len(state) == 54 and all(isinstance(x, str) for x in state):
        return state
    if isinstance(state, torch.Tensor):
        return decode_cube_state(state)
    if isinstance(state, str):
        return list(state)
    # assume it's already an encoded string for encode_cube_state
    return list(state)


def get_cube_child_states(cube):
    children = []
    original_state = cube.state.copy()
    
    for move in cube.moves:
        child_cube = Cube(state=original_state.copy())
        child_cube.move(move)
        child_state = child_cube.state.copy()
        children.append(child_state)
    
    return children

def _model_value_score(state_tensor, model):
   
    try:
        with torch.no_grad():
            policy_logits, value = model(state_tensor)  # value scalar
        v = value.item() if torch.is_tensor(value) else float(value)
        return float(v)
    except Exception:
        return None

def _tensor_from_state_list(state_list):
    enc = encode_cube_state(''.join(state_list))
    return torch.FloatTensor(enc).unsqueeze(0)

def _state_key_from_list(state_list):
    # state_list: ['w','y',...]*54
    return ''.join(state_list)

def dijktras (cube_state):
    s_list = _to_state_list(cube_state)
    root_cube = Cube(state=s_list)
    if root_cube.is_solved():
        return [], True
    
    parent_value=_model_value_score(cube_state, model)
    children_values=[]
    action_path=[]
    children=get_cube_child_states(root_cube)

    for child in children:
        if child.is_solved():
            return action_path
        else:
            child_value=_model_value_score(child, model)
            children_values.append(child_value)

    children_values.sort()

    child[12]=root_cube

    


    
    
