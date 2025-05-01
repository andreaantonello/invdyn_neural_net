import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Model definition (same as training)
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

# === Load saved model
model = GravityNet(input_size=18, output_size=6)
model.load_state_dict(torch.load("gravity_net_model.pth"))
model.eval()

# === Load scalers
scaler_X = {
    'mean': np.load('scaler_X_mean.npy'),
    'scale': np.load('scaler_X_scale.npy'),
}
scaler_Y = {
    'mean': np.load('scaler_Y_mean.npy'),
    'scale': np.load('scaler_Y_scale.npy'),
}

def scale_input(X):
    return (X - scaler_X['mean']) / scaler_X['scale']

def descale_output(Y_scaled):
    return Y_scaled * scaler_Y['scale'] + scaler_Y['mean']

# === Load original CSV (same structure as training)
data = pd.read_csv("input/robot_data.csv", header=None).to_numpy()

# Validate shape
if data.shape[1] != 24:
    raise ValueError(f"Expected 24 columns, found {data.shape[1]}.")

X = data[:, :18]
Y_true = data[:, 18:24]

# === Run model prediction
X_scaled = scale_input(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

with torch.no_grad():
    Y_pred_scaled = model(X_tensor).numpy()

Y_pred = descale_output(Y_pred_scaled)

# === Print first 5 predictions for sanity check
print("\nSample predictions:")
for i in range(5):
    print(f"  True τ: {Y_true[i]}")
    print(f"  Pred τ: {Y_pred[i]}")
    print("---")

# === Compute percent error with a mask to avoid near-zero torque
eps = 1e-8
mask = np.abs(Y_true) > 1e-2
percent_error = 100 * np.abs(Y_true - Y_pred) / (np.abs(Y_true) + eps)
percent_error[~mask] = 0  # Zero out unreliable percent errors

mean_percent_error_per_joint = np.mean(percent_error, axis=0)
overall_mean_percent_error = np.mean(mean_percent_error_per_joint)

# === Compute MAE and RMSE
mae_per_joint = mean_absolute_error(Y_true, Y_pred, multioutput='raw_values')
rmse_per_joint = np.sqrt(mean_squared_error(Y_true, Y_pred, multioutput='raw_values'))

# === Print results
print("\nMean percent error per joint (torque):")
for i, err in enumerate(mean_percent_error_per_joint):
    print(f"  Joint {i+1}: {err:.2f}%")

print(f"\nOverall mean percent error: {overall_mean_percent_error:.2f}%")

print("\nMean Absolute Error (Nm):")
for i, err in enumerate(mae_per_joint):
    print(f"  Joint {i+1}: {err:.4f}")

print("\nRoot Mean Squared Error (Nm):")
for i, err in enumerate(rmse_per_joint):
    print(f"  Joint {i+1}: {err:.4f}")
