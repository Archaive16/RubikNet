import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

from cube import Cube, encode_cube_state, decode_cube_state
from dataset import generate_dataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device available: {device}")


# ADI Neural Network Model
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


def targets(states, pred_vals, device='cpu'):
    # Calculate target policy and value for training for a Base state using the Child states
    
    action_vals = []

    with torch.no_grad():
        for i, state in enumerate(states):
            cube = Cube(state)
            reward = cube.get_reward()
            # Handle tensor on any device
            if isinstance(pred_vals[i], torch.Tensor):
                pred_val = torch.clamp(pred_vals[i].squeeze(), -1, 1).item()
            else:
                pred_val = np.clip(pred_vals[i], -1, 1)

            action_val = reward + pred_val
            action_vals.append(action_val)

        max_i = np.argmax(action_vals)
        target_val = action_vals[max_i]

        target_val = np.tanh(target_val)

        target_policy_i = torch.tensor(max_i, dtype=torch.long, device=device)
        target_val = torch.tensor([target_val], dtype=torch.float32, device=device)

        return target_policy_i, target_val


def prepare_dataloader(df, batch_size, shuffle=True):
    # Prepare DataLoader from DataFrame.
    
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
    # Train the final adi model
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

            # Move batch data to device
            batch_bases = batch_bases.to(device)
            batch_children = batch_children.to(device)
            batch_scramble_steps = batch_scramble_steps.to(device)

            for iteration in range(batch_iterations):
                print(f"  ITERATION {iteration + 1}/{batch_iterations}")

                # Shuffle indices
                shuffle_indices = torch.randperm(len(batch_bases))
                batch_bases = batch_bases[shuffle_indices]
                batch_children = batch_children[shuffle_indices]
                batch_scramble_steps = batch_scramble_steps[shuffle_indices]

                optimizer.zero_grad()

                total_policy_loss = 0
                total_value_loss = 0

                for base_i in range(len(batch_bases)):
                    base = batch_bases[base_i]  # Already on device
                    children_i = batch_children[base_i]  # Already on device
                    depth = batch_scramble_steps[base_i].item()

                    with torch.no_grad():
                        W = 1 / max(depth, 1.0)

                        child_predicted_values = []
                        for child in children_i:
                            _, child_pred_val = model.forward(child.unsqueeze(0))
                            child_predicted_values.append(child_pred_val.squeeze())

                        child_predicted_values_t = torch.stack(child_predicted_values)

                        # Decode children (automatically handles GPU tensors)
                        children_decoded = [decode_cube_state(child) for child in children_i]

                        target_policy_i, target_value = targets(children_decoded, child_predicted_values_t, device=device)

                    # Forward pass for base state
                    predicted_policy, predicted_value = model.forward(base.unsqueeze(0))
                    predicted_policy = predicted_policy.squeeze(0)
                    predicted_value = predicted_value.squeeze()

                    # Calculate losses
                    loss_policy = criterion_policy(predicted_policy.unsqueeze(0), target_policy_i.unsqueeze(0))
                    loss_val = criterion_val(predicted_value.unsqueeze(0), target_value)

                    # Apply weighting
                    weighted_policy_loss = W * loss_policy
                    weighted_value_loss = W * loss_val

                    total_policy_loss += weighted_policy_loss
                    total_value_loss += weighted_value_loss

                # Average losses
                avg_policy_loss = total_policy_loss / len(batch_bases)
                avg_value_loss = total_value_loss / len(batch_bases)
                avg_total_loss = avg_policy_loss + avg_value_loss

                # Backward pass
                avg_total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                # Record losses
                epoch_loss += avg_total_loss.item()
                batch_losses.append(avg_total_loss.item())
                policy_losses.append(avg_policy_loss.item())
                val_losses.append(avg_value_loss.item())
                total_iterations += 1

                print(f"    Loss: {avg_total_loss.item():.6f} (Policy: {avg_policy_loss.item():.6f}, Value: {avg_value_loss.item():.6f})")

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / total_iterations
        epoch_losses.append(avg_epoch_loss)
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.6f}")

        # Clear cache periodically to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save model
    torch.save(model.state_dict(), 'deepcube_adi_model.pth')
    print(f"\nModel saved as 'deepcube_adi_model.pth'")

    return model, batch_losses, epoch_losses, policy_losses, val_losses


def plot_training_results(batch_losses, epoch_losses, policy_losses, val_losses):
    # Plot batch losses
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(batch_losses)+1), batch_losses)
    plt.xlabel("Batch Iteration")
    plt.ylabel("Total Loss")
    plt.title("Total Loss vs Batch Iterations")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Average Loss vs Epochs")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(policy_losses)+1), policy_losses)
    plt.xlabel("Batch Iteration")
    plt.ylabel("Policy Loss")
    plt.title("Policy Loss vs Batch Iterations")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(val_losses)+1), val_losses)
    plt.xlabel("Batch Iteration")
    plt.ylabel("Value Loss")
    plt.title("Value Loss vs Batch Iterations")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def test_model(model_path='deepcube_adi_model.pth', num_tests=10, scramble_depth=5):
    # Test the trained model on scrambled cubes.
    
    model = ADI().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\nTesting model on {num_tests} scrambled cubes with depth {scramble_depth}")

    for test_i in range(num_tests):
        cube = Cube()
        cube.scramble(scramble_depth)

        print(f"\nTest {test_i + 1}:")
        print(f"Scramble moves: {cube.move_history}")

        # Encode state
        state_encoded = encode_cube_state(''.join(cube.state))
        state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0).to(device)

        # Get model predictions
        with torch.no_grad():
            policy_logits, value = model(state_tensor)
            policy_probs = F.softmax(policy_logits, dim=-1)
            predicted_move_idx = torch.argmax(policy_probs).item()
            predicted_value = value.item()

        print(f"Predicted best move: {cube.moves[predicted_move_idx]}")
        print(f"Predicted value: {predicted_value:.4f}")
        print(f"Policy probabilities: {policy_probs.cpu().numpy()}")


if __name__ == "__main__":
    # TRAINING PARAMETERS
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    BATCH_ITERATIONS = 10
    SCRAMBLE_DEPTH = 12
    SCRAMBLE_RUNS_PER_EPOCH = 200

    print("Starting ADI training...")
    print(f"Configuration:")
    print(f"  - Device: {device}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Batch Iterations: {BATCH_ITERATIONS}")
    print(f"  - Scramble Depth: {SCRAMBLE_DEPTH}")
    print(f"  - Scramble Runs per Epoch: {SCRAMBLE_RUNS_PER_EPOCH}")

    # Train the model
    model, batch_losses, epoch_losses, policy_losses, val_losses = train_adi(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        batch_iterations=BATCH_ITERATIONS,
        scramble_depth=SCRAMBLE_DEPTH,
        scramble_runs_per_epoch=SCRAMBLE_RUNS_PER_EPOCH
    )

    print("\nTraining completed!")
    print(f"Final epoch loss: {epoch_losses[-1]:.6f}")

    # Plot results
    plot_training_results(batch_losses, epoch_losses, policy_losses, val_losses)

    # Optional: Test the model
    # test_model('deepcube_adi_model.pth', num_tests=5, scramble_depth=5)