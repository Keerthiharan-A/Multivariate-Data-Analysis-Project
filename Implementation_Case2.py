import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchdiffeq import odeint

torch.manual_seed(42)
np.random.seed(42)

# DYNAMICS FUNCTIONS  
class SpiralFunc(nn.Module):
    """True dynamics — fixed inward spiral matrix"""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(Tensor([[-0.1, -1.], [1., -0.1]]), requires_grad=False)

    def forward(self, t, x):   # torchdiffeq expects (t, x) order
        return self.lin(x)

class LearnedFunc(nn.Module):
    """Randomly initialized linear dynamics to be trained"""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        nn.init.normal_(self.lin.weight, std=0.5)

    def forward(self, t, x):
        return self.lin(x)

#   DATA GENERATION   
z0       = torch.Tensor([[0.6, 0.3]])
t_max    = 6.29 * 5
n_points = 200

times_np = np.linspace(0, t_max, num=n_points)
times    = torch.from_numpy(times_np).float()

with torch.no_grad():
    obs = odeint(SpiralFunc(), z0, times)          # shape: (200, 1, 2)
    obs = obs + torch.randn_like(obs) * 0.01

index_np = np.arange(n_points)[:, None]
times_np = times_np[:, None]

#   BATCH CREATION    
min_delta_time = 1.0
max_delta_time = 5.0
max_points_num = 32

def create_batch():
    t0  = np.random.uniform(0, t_max - max_delta_time)
    t1  = t0 + np.random.uniform(min_delta_time, max_delta_time)
    idx = sorted(np.random.permutation(
        index_np[(times_np > t0) & (times_np < t1)]
    )[:max_points_num])
    t_batch = times[idx]
    return obs[idx], t_batch

#   TRAINING       
func      = LearnedFunc()
optimizer = torch.optim.Adam(func.parameters(), lr=0.01)
n_steps   = 5000
losses    = []

print("Training Neural ODE — Spiral Dynamics Learning")
print(f"True W = [[-0.1, -1.0], [1.0, -0.1]] | Adam lr=0.01 | {n_steps} steps\n")

for i in range(n_steps):
    obs_, ts_ = create_batch()
    z_   = odeint(func, obs_[0], ts_)             # (time, batch, dim)
    loss = F.mse_loss(z_, obs_)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if i % 100 == 0:
        print(f"Step {i:4d} | Loss: {loss.item():.6e}")

print(f"Step {n_steps-1:4d} | Loss: {losses[-1]:.6e}")

#   EVALUATION      
with torch.no_grad():
    z_pred = odeint(func, z0, times)

obs_np  = obs.squeeze(1).numpy()
pred_np = z_pred.squeeze(1).numpy()
t_np    = times.numpy()

mse  = np.mean((pred_np - obs_np) ** 2)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(pred_np - obs_np))
r2   = 1 - np.sum((obs_np - pred_np)**2) / np.sum((obs_np - np.mean(obs_np, axis=0))**2)
maxe = np.max(np.abs(pred_np - obs_np))

true_W    = np.array([[-0.1, -1.0], [1.0, -0.1]])
learned_W = func.lin.weight.detach().numpy()

print(f"\n--- Quantitative Results ---")
print(f"  MSE: {mse:.4e} | RMSE: {rmse:.4e} | MAE: {mae:.4e} | R²: {r2:.4f} | MaxErr: {maxe:.4e}")
print(f"\n  True W    :\n{true_W}")
print(f"\n  Learned W :\n{learned_W.round(4)}")

#   PLOTS        
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Neural ODE — Spiral Dynamics Learning", fontsize=13, fontweight='bold')

ax = axes[0]
sc = ax.scatter(obs_np[:, 0], obs_np[:, 1], c=t_np, cmap=cm.plasma, s=8, label="Noisy obs", zorder=3)
ax.plot(pred_np[:, 0], pred_np[:, 1], 'b-', lw=2, alpha=0.8, label="Learned traj")
plt.colorbar(sc, ax=ax, label="Time")
ax.set_title("Phase Portrait (x₁ vs x₂)")
ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(t_np, obs_np[:, 0],  color='lightsteelblue', lw=1.2, label="Obs x₁")
ax.plot(t_np, obs_np[:, 1],  color='lightsalmon',    lw=1.2, label="Obs x₂")
ax.plot(t_np, pred_np[:, 0], color='steelblue',  lw=2, label="Pred x₁")
ax.plot(t_np, pred_np[:, 1], color='darkorange', lw=2, label="Pred x₂")
ax.set_title("State Trajectories Over Time")
ax.set_xlabel("Time"); ax.set_ylabel("State"); ax.legend(fontsize=8)

ax = axes[2]
labels    = ['W[0,0]', 'W[0,1]', 'W[1,0]', 'W[1,1]']
true_vals = [-0.1, -1.0, 1.0, -0.1]
pred_vals = [learned_W[0,0], learned_W[0,1], learned_W[1,0], learned_W[1,1]]
x = np.arange(len(labels)); w = 0.35
ax.bar(x - w/2, true_vals, w, label='True W',    color='steelblue',  alpha=0.8)
ax.bar(x + w/2, pred_vals, w, label='Learned W', color='darkorange', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.axhline(0, color='black', lw=0.8)
ax.set_title("True vs Learned Weight Matrix")
ax.set_ylabel("Value"); ax.legend()

plt.tight_layout()
plt.savefig("spiral_results.png", dpi=150, bbox_inches='tight')
plt.show()