import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import csv
import pandas as pd
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

# CUBE AGENT/CUBE MODEL
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

    def move(self, move_name):
        self.move_history.append(move_name)
        self.state = self.switch(move_name, self.state)

    def print_cube(self):
        print("F:", self.state[0:9])
        print("B:", self.state[9:18])
        print("U:", self.state[18:27])
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

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device available ", device)

class ADI(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(324, 4096)
        self.fc2 = nn.Linear(4096, 2048)

        # Policy Head
        self.fc3_1 = nn.Linear(2048, 512)
        self.fc_policy_out = nn.Linear(512, 12)

        # Value Head
        self.fc3_2 = nn.Linear(2048, 512)
        self.fc_value_out = nn.Linear(512, 1)

        self.apply_glorot_init()

    def apply_glorot_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        # Policy Path
        policy_logits = F.elu(self.fc3_1(x))
        policy_logits = self.fc_policy_out(policy_logits)

        # Value Path
        value = F.elu(self.fc3_2(x))
        value = torch.tanh(self.fc_value_out(value))

        return policy_logits, value

# GENERATING DATASET
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

# FUNCTIONS TO ENCODE AND DECODE STATES
color_map = {
    'w': [1, 0, 0, 0, 0, 0],  # White
    'y': [0, 1, 0, 0, 0, 0],  # Yellow
    'b': [0, 0, 1, 0, 0, 0],  # Blue
    'g': [0, 0, 0, 1, 0, 0],  # Green
    'r': [0, 0, 0, 0, 1, 0],  # Red
    'o': [0, 0, 0, 0, 0, 1]   # Orange
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
        encoded_state = encoded_state.detach().numpy()

    for i in range(0, len(encoded_state), 6):
        color_vec = encoded_state[i:i+6]
        color_i = np.argmax(color_vec)
        decoded.append(color_list[color_i])
    return decoded

def targets(states, pred_vals):
    action_vals = []

    with torch.no_grad():
        for i, state in enumerate(states):
            cube = Cube(state)
            reward = cube.get_reward()
            pred_val = torch.clamp(pred_vals[i].squeeze(), -1, 1).item()

            action_val = reward + pred_val
            action_vals.append(action_val)

        max_i = np.argmax(action_vals)
        target_val = action_vals[max_i]

        target_val = np.tanh(target_val)

        target_policy_i = torch.tensor(max_i, dtype=torch.long)
        target_val = torch.tensor([target_val], dtype=torch.float32)

        return target_policy_i, target_val

def prepare_dataloader(df, batch_size, shuffle=True):
    child_encoded = df['child_state'].apply(encode_cube_state)
    child_encoded = np.array(child_encoded.tolist())
    children = torch.FloatTensor(child_encoded)

    base_states = df['base_state'].values[::12]
    base_encoded = [encode_cube_state(state) for state in base_states]
    bases = torch.FloatTensor(np.array(base_encoded))

    children = children.view(-1, 12, children.size(-1))

    scramble_steps = df['scramble_step'].values[::12]
    scramble_steps = torch.tensor(scramble_steps, dtype=torch.long)

    dataset = TensorDataset(bases, children, scramble_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def train_adi(num_epochs, batch_size, batch_iterations=1, scramble_depth=5, scramble_runs_per_epoch=50):

    model = ADI().to(device)

    policy_params = list(model.fc3_1.parameters()) + list(model.fc_policy_out.parameters())
    value_params = list(model.fc3_2.parameters()) + list(model.fc_value_out.parameters())
    shared_params = list(model.fc1.parameters()) + list(model.fc2.parameters())

    optimizer = optim.AdamW([
        {'params': policy_params, 'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': value_params, 'lr': 5e-5, 'weight_decay': 1e-4},
        {'params': shared_params, 'lr': 1e-5, 'weight_decay': 5e-5}
    ])

    criterion_policy = nn.CrossEntropyLoss()
    criterion_val = nn.MSELoss()

    epoch_losses = []
    batch_losses = []
    policy_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch + 1}/{num_epochs}")

        print("Generating new dataset...")
        generate_dataset(scramble_depth, scramble_runs_per_epoch, f"cube_dataset_epoch_{epoch+1}.csv")

        df = pd.read_csv(f"cube_dataset_epoch_{epoch+1}.csv")

        dataloader = prepare_dataloader(df, batch_size, shuffle=True)

        print(f"Dataset loaded. DataLoader created with {len(dataloader)} batches.")

        epoch_loss = 0
        total_iterations = 0

        for batch_i, batch_data in enumerate(dataloader):
            print(f"BATCH {batch_i + 1}/{len(dataloader)}")

            batch_bases, batch_children, batch_scramble_steps = batch_data

            for iteration in range(batch_iterations):
                print(f"  ITERATION {iteration + 1}/{batch_iterations}")

                shuffle_indices = torch.randperm(len(batch_bases))
                batch_bases = batch_bases[shuffle_indices]
                batch_children = batch_children[shuffle_indices]
                batch_scramble_steps = batch_scramble_steps[shuffle_indices]

                optimizer.zero_grad()

                total_policy_loss = 0
                total_value_loss = 0

                for base_i in range(len(batch_bases)):
                    base = batch_bases[base_i]
                    children_i = batch_children[base_i]
                    depth = batch_scramble_steps[base_i].item()

                    with torch.no_grad():
                        W = 1 / max(depth, 1.0)

                        child_predicted_values = []
                        for child in children_i:
                            _, child_pred_val = model.forward(child.unsqueeze(0))
                            child_predicted_values.append(child_pred_val.squeeze())

                        child_predicted_values_t = torch.stack(child_predicted_values)

                        children_decoded = [decode_cube_state(child) for child in children_i]

                        target_policy_i, target_value = targets(children_decoded, child_predicted_values_t)

                    predicted_policy, predicted_value = model.forward(base.unsqueeze(0))
                    predicted_policy = predicted_policy.squeeze(0)
                    predicted_value = predicted_value.squeeze()

                    loss_policy = criterion_policy(predicted_policy.unsqueeze(0), target_policy_i.unsqueeze(0))
                    loss_val = criterion_val(predicted_value.unsqueeze(0), target_value)

                    weighted_policy_loss = W * loss_policy
                    weighted_value_loss = W * loss_val

                    total_policy_loss += weighted_policy_loss
                    total_value_loss += weighted_value_loss

                avg_policy_loss = total_policy_loss / len(batch_bases)
                avg_value_loss = total_value_loss / len(batch_bases)
                avg_total_loss = avg_policy_loss + avg_value_loss

                avg_total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                epoch_loss += avg_total_loss.item()
                batch_losses.append(avg_total_loss.item())
                policy_losses.append(avg_policy_loss.item())
                val_losses.append(avg_value_loss.item())
                total_iterations += 1

                print(f"    Loss: {avg_total_loss.item()} (Policy: {avg_policy_loss.item()}, Value: {avg_value_loss.item()})")

        avg_epoch_loss = epoch_loss / total_iterations
        epoch_losses.append(avg_epoch_loss)
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss}")

    torch.save(model.state_dict(), 'deepcube_adi_model.pth')
    print(f"\nModel saved as 'deepcube_adi_model.pth'")

    return model, batch_losses, epoch_losses, policy_losses, val_losses

def plot_training_results(batch_losses, epoch_losses, policy_losses, val_losses):

    plt.plot(range(1, len(batch_losses)+1), batch_losses)
    plt.xlabel("Batch")
    plt.ylabel("Batch Loss")
    plt.title("Batch Loss vs Steps")
    plt.show()

    plt.plot(range(1, len(epoch_losses)+1), epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Epoch Loss")
    plt.title("Average Loss vs Epochs")
    plt.show()

    plt.plot(range(1, len(policy_losses)+1), policy_losses)
    plt.xlabel("Batch")
    plt.ylabel("Policy Loss")
    plt.title("Policy Loss vs Steps")
    plt.show()

    plt.plot(range(1, len(val_losses)+1), val_losses)
    plt.xlabel("Batch")
    plt.ylabel("Value Loss")
    plt.title("Value Loss vs Steps")
    plt.show()

if __name__ == "__main__":
    # DEFINE
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    BATCH_ITERATIONS = 10
    SCRAMBLE_DEPTH = 12
    SCRAMBLE_RUNS_PER_EPOCH = 200

    print("Starting ADI training...")

    model, batch_losses, epoch_losses, policy_losses, val_losses = train_adi(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        batch_iterations=BATCH_ITERATIONS,
        scramble_depth=SCRAMBLE_DEPTH,
        scramble_runs_per_epoch=SCRAMBLE_RUNS_PER_EPOCH
    )

    print("\nTraining completed!")
    print(f"Final epoch loss: {epoch_losses[-1]}")

    plot_training_results(batch_losses, epoch_losses, policy_losses, val_losses)