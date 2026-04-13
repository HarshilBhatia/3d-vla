import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ============================================================
# Config
# ============================================================
dim = 16              # head dim → 8 frequency bins
base = 10
num_pairs = 32        # number of frozen (Q, K) pairs
m_star = 5.0          # ground truth position
m_init = 4.5          # starting guess
lr = 0.01
steps = 800

# Per-bin gradient control: 1 = allow gradients, 0 = block gradients
# Example: only allow gradients from bins 2, 3, 4
# bin_gradient_mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
bin_gradient_mask = torch.tensor([1,1,1,1, 1, 1, 1, 1], dtype=torch.float32)

# ============================================================
# Frozen Q, K pairs
# ============================================================
Qs = torch.randn(num_pairs, dim)   # [N, dim]
Ks = torch.randn(num_pairs, dim)   # [N, dim]

# ============================================================
# RoPE helpers
# ============================================================
def get_freqs(dim, base):
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

def apply_rope(x, m, freqs, bin_mask=None):
    """
    x:        [N, dim] or [dim]
    m:        scalar
    freqs:    [dim//2]
    bin_mask: [dim//2] or None. If provided, masks which bins' gradients can flow.
              1.0 = allow gradients, 0.0 = block gradients for that bin.
    """
    if bin_mask is not None:
        # For bins with mask=0, detach m so gradients don't flow through them
        # For bins with mask=1, keep m differentiable
        angles = torch.zeros_like(freqs)
        for i in range(len(freqs)):
            if bin_mask[i] > 0.5:
                # Allow gradients
                angles[i] = m * freqs[i]
            else:
                # Block gradients by detaching m for this bin
                angles[i] = m.detach() * freqs[i]
    else:
        angles = m * freqs                          # [dim//2]
    
    cos_a = angles.cos()
    sin_a = angles.sin()
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.zeros_like(x)
    out[..., 0::2] = x1 * cos_a - x2 * sin_a
    out[..., 1::2] = x1 * sin_a + x2 * cos_a
    return out

freqs = get_freqs(dim, base)
print("Frequency bins (high → low freq):", [f"{f:.6f}" for f in freqs.tolist()])
print("Bin gradient mask:", bin_gradient_mask.tolist())
print("Active bins:", [i for i, m in enumerate(bin_gradient_mask) if m > 0.5])

# ============================================================
# Targets: dot(RoPE(Q_i, m*), K_i) for each pair
# ============================================================
with torch.no_grad():
    Qs_rot_star = apply_rope(Qs, torch.tensor(m_star), freqs)  # [N, dim]
    targets = (Qs_rot_star * Ks).sum(dim=-1)                   # [N]

print(f"m* = {m_star}, m_init = {m_init}\n")

# ============================================================
# Training loop
# ============================================================
m = torch.tensor(m_init, requires_grad=True)

loss_history = []
m_history = []
grad_history = []
per_bin_grad_history = []

for step in range(steps):
    # Forward: compute all dot products at current m
    # Apply bin_gradient_mask to control which bins contribute gradients
    Qs_rot = apply_rope(Qs, m, freqs, bin_mask=bin_gradient_mask)  # [N, dim]
    preds = (Qs_rot * Ks).sum(dim=-1)              # [N]
    loss = ((preds - targets) ** 2).mean()          # scalar

    # Backward (only gradients from masked bins will flow)
    loss.backward()

    # --- Decompose gradient into per-bin contributions ---
    with torch.no_grad():
        per_bin_grads = []
        for i in range(len(freqs)):
            masked_freqs = torch.zeros_like(freqs)
            masked_freqs[i] = freqs[i]

            # Predictions using only bin i's rotation
            Qs_rot_i = apply_rope(Qs, m.detach(), masked_freqs)
            preds_i = (Qs_rot_i * Ks).sum(dim=-1)

            # Finite diff for d(preds)/dm at bin i
            eps = 1e-5
            Qs_rot_i_plus = apply_rope(Qs, m.detach() + eps, masked_freqs)
            preds_i_plus = (Qs_rot_i_plus * Ks).sum(dim=-1)
            dpreds_dm_i = (preds_i_plus - preds_i) / eps      # [N]

            # Chain rule: d(loss)/dm contribution from bin i
            # loss = mean((preds - targets)^2)
            # d(loss)/dm = mean(2*(preds - targets) * dpreds/dm)
            # residuals use the FULL pred (all bins), dpreds/dm is bin i's contribution
            residuals = (preds - targets)                       # [N]
            bin_grad = (2 * residuals * dpreds_dm_i).mean().item()
            per_bin_grads.append(bin_grad)

    # Log
    loss_history.append(loss.item())
    m_history.append(m.item())
    grad_history.append(m.grad.item())
    per_bin_grad_history.append(per_bin_grads)

    # Raw SGD — no Adam, no momentum
    with torch.no_grad():
        m -= lr * m.grad
    m.grad.zero_()

# ============================================================
# Plotting
# ============================================================
per_bin_grad_history = torch.tensor(per_bin_grad_history)  # [steps, num_bins]

# Downsample for cleaner plots (plot every Nth point)
plot_step = 5  # Plot every 5th point
plot_indices = range(0, len(m_history), plot_step)

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

# 1) m over time
m_plot = [m_history[i] for i in plot_indices]
axes[0].plot(plot_indices, m_plot, color="steelblue", linewidth=1.5, marker='o', markersize=3)
axes[0].axhline(m_star, color="red", linestyle="--", label=f"m* = {m_star}")
axes[0].set_ylabel("m")
axes[0].set_title("Learned position m over training")
axes[0].legend()

# 2) Loss (log scale)
loss_plot = [loss_history[i] for i in plot_indices]
axes[1].semilogy(plot_indices, loss_plot, color="darkorange", linewidth=1.5, marker='o', markersize=3)
axes[1].set_ylabel("Loss")
axes[1].set_title("Loss over training")

# 3) Per-bin gradient contributions (only show active bins)
num_bins = len(freqs)
active_bins = [i for i in range(num_bins) if bin_gradient_mask[i] > 0.5]
colors = plt.cm.viridis(torch.linspace(0, 1, len(active_bins)).numpy())

for idx, i in enumerate(active_bins):
    label = f"bin {i} (θ={freqs[i].item():.4f})"
    grad_plot = per_bin_grad_history[plot_indices, i].numpy()
    axes[2].plot(plot_indices, grad_plot, color=colors[idx], 
                 linewidth=1.5, marker='o', markersize=3, label=label)

axes[2].axhline(0, color="gray", linestyle="-", linewidth=0.5)
axes[2].set_ylabel("Gradient contribution")
axes[2].set_xlabel("Step")
axes[2].set_title(f"Per-bin gradient contributions (active bins: {active_bins})")
axes[2].legend(fontsize=9, loc="best")

plt.tight_layout()
plt.savefig("rope_grad_toy_last4.png", dpi=150)
plt.show()
print("Done. Check rope_grad_toy.png")