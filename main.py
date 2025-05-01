import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim import RAdam
from torch.utils.data import DataLoader, TensorDataset
import time

# === Load and Check CSV ===
csv_file = 'input/robot_data.csv'
df = pd.read_csv(csv_file, header=None)
data = df.to_numpy()

expected_cols = 24
if data.shape[1] != expected_cols:
    raise ValueError(f"Expected {expected_cols} columns, but got {data.shape[1]}")

# === Split Inputs and Outputs
X = data[:, :18]
Y = data[:, 18:24]

scaler_X = StandardScaler().fit(X)
scaler_Y = StandardScaler().fit(Y)

X_scaled = scaler_X.transform(X)
Y_scaled = scaler_Y.transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Convert to tensors and create DataLoader for batch processing
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # smaller batch size

# === Define Network with Dropout for Regularization ===
class GravityNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(GravityNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.net(x)

model = GravityNet(input_size=18, output_size=6)

# === Optimizer and Scheduler ===
optimizer = RAdam(model.parameters(), lr=1e-3)  # Switch to RAdam
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)  # Adaptive learning rate

# === Training Loop with Early Stopping and Plotting ===
epochs = 3000  # Increase epochs
patience = 50  # Decreased patience
best_val_loss = float('inf')
early_stop_counter = 0
train_losses = []
val_losses = []

plt.ion()  # interactive mode on
fig, ax = plt.subplots()

start_time = time.time()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for i, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = nn.MSELoss()(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Average epoch loss
    avg_loss = epoch_loss / len(train_loader)
    model.eval()
    val_loss = nn.MSELoss()(model(X_test_tensor), Y_test_tensor).item()
    scheduler.step(val_loss)  # Step the scheduler based on validation loss

    train_losses.append(avg_loss)
    val_losses.append(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print(f"Early stopping after {epoch+1} epochs due to no improvement in validation loss.")
        break

    # Update plot
    ax.clear()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_title("Training Progress")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.pause(0.01)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f}s")

# === Save model and scalers ===
torch.save(model.state_dict(), 'gravity_net_model.pth')
np.save('scaler_X_mean.npy', scaler_X.mean_)
np.save('scaler_X_scale.npy', scaler_X.scale_)
np.save('scaler_Y_mean.npy', scaler_Y.mean_)
np.save('scaler_Y_scale.npy', scaler_Y.scale_)
print("Model and scalers saved.")

plt.ioff()
plt.show()

# === Predict and Plot Torque Computation Performance ===

# Get predictions on the test set
model.eval()
with torch.no_grad():
    Y_pred_scaled = model(X_test_tensor)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled.numpy())

# Plot predicted vs actual torque values
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(6):
    axes[i].scatter(Y_test[:, i], Y_pred[:, i], color='blue', label='Predicted vs Actual')
    axes[i].plot([min(Y_test[:, i]), max(Y_test[:, i])], [min(Y_test[:, i]), max(Y_test[:, i])], color='red', linestyle='--', label='Perfect Prediction')
    axes[i].set_title(f'Torque {i+1}')
    axes[i].set_xlabel('Actual Torque')
    axes[i].set_ylabel('Predicted Torque')
    axes[i].legend()

plt.tight_layout()
plt.show()
