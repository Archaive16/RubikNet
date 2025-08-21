import torch
from adi import ADI
from cube import Cube

model = ADI()
model.load_state_dict(torch.load("deepcube_adi_model.pth", map_location='cpu'))

model.eval()

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




import numpy as np
import torch

# ------- small helpers -------

def _state_key_from_list(state_list):
    # state_list: ['w','y',...]*54
    return ''.join(state_list)

def _to_state_list(state):
    
    if isinstance(state, list) and len(state) == 54 and all(isinstance(x, str) for x in state):
        return state
    if isinstance(state, torch.Tensor):
        return decode_cube_state(state)
    if isinstance(state, str):
        return list(state)
    # assume it's already an encoded string for encode_cube_state
    return list(state)

def _inverse_move(m):
    
    return m[:-1] if m.endswith("'") else m + "'"

def _heuristic_by_centers(state_list):
    
    if not isinstance(state_list, list) or len(state_list) != 54:
        return 0.0
    score = 0
    for f in range(6):
        base = f * 9
        center = state_list[base + 4]
        face = state_list[base:base + 9]
        score += sum(1 for c in face if c == center)
    return float(score)

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

# ------- main algorithm -------

def beam_search(
    start_state,
    beam_width=5,
    max_depth=20,
    model=None,
    prune_inverses=True,
    avoid_repeats=True
):
    """
    Beam search for Rubik's Cube.
    Returns (moves_list, solved_bool).

    - start_state: list[str] of len 54 OR encoded tensor/string; we normalize it.
    - beam_width: number of nodes kept per depth.
    - max_depth: maximum search depth.
    - model: optional; if provided, the model's value head is blended into scoring.
    - prune_inverses: avoid immediately undoing the last move.
    - avoid_repeats: drop children that produce states already seen at any shallower depth.
    """

   
    s_list = _to_state_list(start_state)
    root_cube = Cube(state=s_list)
    if root_cube.is_solved():
        return [], True

   
    def node_score(state_list):
        # Blend heuristic with model value if available
        base = _heuristic_by_centers(state_list)  # 0..54
        if model is not None:
            st = _tensor_from_state_list(state_list)
            mv = _model_value_score(st, model)
        else:
            mv = None
        if mv is None:
            return base
       
        return base + 50.0 * mv

    
    start_key = _state_key_from_list(s_list)
    beam = [(s_list, [], None, node_score(s_list))]
    seen = {start_key: 0}  # state_key -> best (smallest) depth seen

    for depth in range(1, max_depth + 1):
        candidates = []

        
        for state_list, path, last_move, _ in beam:
            cube = Cube(state=state_list)
            child_states = get_cube_child_states(cube)  # list of 54-char lists

            
            moves = getattr(cube, "moves", list(range(len(child_states))))

            for idx, child in enumerate(child_states):
                if not isinstance(child, list) or len(child) != 54:
                    continue
                move = moves[idx] if idx < len(moves) else idx

                
                if prune_inverses and last_move is not None:
                    if isinstance(move, str) and _inverse_move(move) == last_move:
                        continue

                key = _state_key_from_list(child)
                if avoid_repeats and key in seen and seen[key] <= depth:
                    continue  # already found at equal/shallower depth

                # solved?
                try:
                    child_cube = Cube(state=child)
                    if child_cube.is_solved():
                        return path + [move], True
                except Exception:
                    # if Cube ctor fails, skip this child
                    continue

                #
                sc = node_score(child)
                candidates.append((child, path + [move], move, sc))
                if avoid_repeats:
                    seen[key] = depth

        if not candidates:
            
            return [], False

        
        candidates.sort(key=lambda x: x[3], reverse=True)
        beam = candidates[:beam_width]

    # depth limit reached with no solution
    return [], False


solved = 0
num_tests = 1  # change to 20 if you want

for i in range(num_tests):
    cube = Cube()
    cube.reset_cube()
    cube.scramble(7)   # scramble with 3 random moves
    print(f"\nTest {i+1}: Scramble moves ->", cube.move_history)

    if cube.is_solved():
        print("Cube was already solved after scramble, skipping...")
        continue

    # Run beam search
    action_path, is_solved = beam_search(
        start_state=cube.state,
        beam_width=200,
        max_depth=50,
        model=model,               
        prune_inverses=True,
        avoid_repeats=True
    )

    print("Beam search output:", action_path)

    if is_solved and action_path:
        # Verify by applying the moves
        test_cube = Cube(state=cube.state.copy())
        for move in action_path:
            test_cube.move(move)
            print("Applied:", move)

        if test_cube.is_solved():
            print("Solved")
            solved += 1
        else:
            print(" Not solved after appl")

