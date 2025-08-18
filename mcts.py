## MCTS

model = ADI()
model.load_state_dict(torch.load("deepcube_adi_model.pth"))
model.eval()

c = 4
v = 150

def get_or_create_node(state, parent = None):
    # Convert tensor to hashable key
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


    def __init__(self, state, parent = None):

        self.children_nodes = []
        self.all_A = []
        self.N = 0
        self.W = 0
        self.L = 0
        self.P = 0
        self.parent = parent
        policy_logits, value = model(state)                  ## state should be encoded and in tensor form
        self.policy_logits = policy_logits
        self.value = value

        decoded_state = decode_cube_state(state)
        self.cube = Cube(decoded_state)

        self.update(self.value)

        self.is_solved = self.cube.is_solved()


    def selection(self):
        """
        A = argmax(U + Q)
        U = cP(s')root(v_c)/(1 + N(s'))
        Q = W - L
        L = L + v (remaining)
        """
        self.N += 1
        self.L += v
        A_index = np.argmax(self.all_A)
        return A_index


    def update(self, value):
        if self.parent is not None:
            self.parent.W = max(self.parent.W, self.value)
            self.parent.update(self.value)
            move_i = find_action_index(self.parent.state, self.state)
            self.P = self.parent.policy_logits[move_i]

        sum_N = 0

        if self.children_nodes:
            for child in self.children_nodes:
                sum_N += child.N

        self.U = (c * self.P * (sum_N ** 0.5)) / (1 + self.N)
        self.Q = self.W - self.L
        self.total = self.Q + self.U

    def expand(self):

        children = self.cube.get_child_states()
        children_encoded = [encode_cube_state(child) for child in children]
        children_tensor = [torch.FloatTensor(child_encoded).unsqueeze(0) for child_encoded in children_encoded]

        for child in children_tensor:
            child_node = get_or_create_node(child, self)
            self.children_nodes.append(child_node)
            self.all_A.append(child_node.total)


def mcts(state):

    state_encoded = encode_cube_state(''.join(state))
    state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0)

    node = NODE(state_tensor)
    is_solved = node.is_solved

    max_steps = 100

    action_path = []

    while not is_solved and max_steps:
        node.expand()
        action_index = node.selection()
        action_path.append(action_index)

        node = node.children_nodes[action_index]
        is_solved = node.is_solved

        max_steps -= 1

    return action_path, is_solved

# random state
this_cube = Cube()
this_cube.reset_cube()
this_cube.scramble(2)

mcts(this_cube.state)

