import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

torch.manual_seed(0)

# TRUE DATA (cos, sin)
t = torch.linspace(0, 10, 200)
x = torch.cos(t)
v = -torch.sin(t)

y_true = torch.stack([x, v], dim=1)

# NEURAL ODE FUNCTION
class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t, y):
        return self.net(y)

func = ODEFunc()

# TRAINING
optimizer = optim.Adam(func.parameters(), lr=1e-3)

for epoch in range(1000):
    y0 = y_true[0].unsqueeze(0)

    pred = odeint(func, y0, t).squeeze(1)

    loss = ((pred - y_true)**2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.6f}")

# EVALUATION
with torch.no_grad():
    pred = odeint(func, y_true[0].unsqueeze(0), t).squeeze(1)

# PLOTS

# Phase space
plt.figure(figsize=(6,6))
plt.plot(y_true[:,0], y_true[:,1], label="True")
plt.plot(pred[:,0], pred[:,1], '--', label="Neural ODE")
plt.legend()
plt.title("Phase Space (x vs v)")
plt.show()

# Time series
plt.figure(figsize=(10,4))
plt.plot(t, y_true[:,0], label="True x")
plt.plot(t, pred[:,0], '--', label="Pred x")
plt.legend()
plt.title("Position vs Time")
plt.show()

y_true_np = y_true.numpy()
pred_np = pred.numpy()

# MSE
mse = np.mean((pred_np - y_true_np)**2)
# RMSE
rmse = np.sqrt(mse)
# MAE
mae = np.mean(np.abs(pred_np - y_true_np))
# R^2 Score
ss_res = np.sum((y_true_np - pred_np)**2)
ss_tot = np.sum((y_true_np - np.mean(y_true_np, axis=0))**2)
r2 = 1 - ss_res/ss_tot

# Max error (important for dynamics)
max_err = np.max(np.abs(pred_np - y_true_np))

print("\n--- Quantitative Results ---")
print(f"MSE   : {mse:.6e}")
print(f"RMSE  : {rmse:.6e}")
print(f"MAE   : {mae:.6e}")
print(f"R^2   : {r2:.6f}")
print(f"Max Error : {max_err:.6e}")