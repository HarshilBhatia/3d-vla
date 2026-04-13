import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ============================================================
# Config
# ============================================================
dim = 16              # head dim → 8 frequency bins (indices 0..7)
base = 10000
num_pairs = 32
m_star = 10.0
m_init = 0.0
lr = 10
steps = 2000

GROUPS = {
    "All bins (0-7)":       [0, 1, 2, 3, 4, 5, 6, 7],
    "High freq (bins 0-2)": [0, 1, 2],
    "Mid freq  (bins 3-4)": [3, 4],
    "Low freq  (bins 5-7)": [5, 6, 7],
}

group_colors = {
    "All bins (0-7)":       "black",
    "High freq (bins 0-2)": "crimson",
    "Mid freq  (bins 3-4)": "goldenrod",
    "Low freq  (bins 5-7)": "steelblue",
}

# ============================================================
# Frozen Q vectors
# ============================================================
Qs = torch.randn(num_pairs, dim)

# ============================================================
# RoPE helpers
# ============================================================
def get_freqs(dim, base):
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

freqs = get_freqs(dim, base)
print("Freqs:", [f"{f:.6f}" for f in freqs.tolist()])
print(f"Rotations over [0, m*={m_star}]:", [f"{(m_star*f/(2*3.14159)):.2f}" for f in freqs.tolist()])

def apply_rope(x, m, freqs):
    """Standard RoPE — all bins use same m, gradient flows through all."""
    angles = m * freqs
    cos_a = angles.cos()
    sin_a = angles.sin()
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.zeros_like(x)
    out[..., 0::2] = x1 * cos_a - x2 * sin_a
    out[..., 1::2] = x1 * sin_a + x2 * cos_a
    return out

def apply_rope_grad_masked(x, m, freqs, active_bins):
    """
    Full RoPE forward (all bins rotate), but only active_bins
    let gradient flow back to m. Blocked bins use m.detach().
    
    This means:
      - Forward is identical to standard RoPE
      - Target uses all bins → loss CAN reach zero at m = m*
      - But d(loss)/dm only gets signal from active_bins
    """
    x1 = x[..., 0::2]   # [N, dim//2]
    x2 = x[..., 1::2]
    out = torch.zeros_like(x)

    for i in range(len(freqs)):
        # Key: active bins use m (grad flows), blocked bins use m.detach() (no grad)
        m_i = m if i in active_bins else m.detach()
        angle = m_i * freqs[i]
        cos_a = angle.cos()
        sin_a = angle.sin()
        out[..., 2*i]   = x1[..., i] * cos_a - x2[..., i] * sin_a
        out[..., 2*i+1] = x1[..., i] * sin_a + x2[..., i] * cos_a

    return out

def make_active_freqs(active_bins):
    af = torch.zeros_like(freqs)
    for b in active_bins:
        af[b] = freqs[b]
    return af

# ============================================================
# Figure 1: Loss landscape (full RoPE target, sweep m)
# ============================================================
m_range = torch.linspace(0, m_star + 10, 5000)
landscapes = {}

with torch.no_grad():
    # Full target (all bins)
    full_target = apply_rope(Qs, torch.tensor(m_star), freqs)

    for name, bins in GROUPS.items():
        af = make_active_freqs(bins)
        # Isolated target: only those bins
        tgt_isolated = apply_rope(Qs, torch.tensor(m_star), af)
        losses_isolated = []
        losses_masked = []
        for m_val in m_range:
            # Isolated: both forward and target use only active bins
            rot_isolated = apply_rope(Qs, m_val, af)
            losses_isolated.append(((rot_isolated - tgt_isolated) ** 2).mean().item())

            # Masked: forward and target both use ALL bins
            rot_full = apply_rope(Qs, m_val, freqs)
            losses_masked.append(((rot_full - full_target) ** 2).mean().item())

        landscapes[name] = {"isolated": losses_isolated, "masked": losses_masked}

fig1, axes1 = plt.subplots(len(GROUPS), 1, figsize=(12, 11), sharex=True)
fig1.suptitle(f"Loss landscape: loss vs m | m*={m_star}", fontsize=13)

for idx, name in enumerate(GROUPS):
    ax = axes1[idx]
    ax.plot(m_range.numpy(), landscapes[name]["isolated"],
            color=group_colors[name], linewidth=1, label="Isolated (own target)")
    # The "masked" landscape is the same for all groups (full RoPE loss)
    # so just plot it once as a reference
    ax.plot(m_range.numpy(), landscapes[name]["masked"],
            color="gray", linewidth=0.6, linestyle="--", alpha=0.5, label="Full RoPE loss (shared)")
    ax.axvline(m_star, color="green", linestyle="--", linewidth=1, label=f"m* = {m_star}")
    ax.axvline(m_init, color="purple", linestyle=":", linewidth=1, label=f"m_init = {m_init}")
    ax.set_ylabel("Loss")
    ax.set_title(name)
    ax.legend(fontsize=7, loc="upper right")

axes1[-1].set_xlabel("m")
plt.tight_layout()
plt.savefig("rope_loss_landscape.png", dpi=150)
plt.show()

# ============================================================
# Training: Run 1 — Isolated (each group has its own target)
# ============================================================
def train_isolated(active_bins, label):
    m = torch.tensor(m_init, requires_grad=True)
    active_freqs = make_active_freqs(active_bins)

    with torch.no_grad():
        targets = apply_rope(Qs, torch.tensor(m_star), active_freqs)

    loss_history, m_history, per_bin_grad_history = [], [], []

    for step in range(steps):
        Qs_rot = apply_rope(Qs, m, active_freqs)
        loss = ((Qs_rot - targets) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            per_bin_grads = []
            for i in range(len(freqs)):
                if i not in active_bins:
                    per_bin_grads.append(0.0)
                    continue
                masked_freqs = torch.zeros_like(freqs)
                masked_freqs[i] = freqs[i]
                eps = 1e-5
                r1 = apply_rope(Qs, m.detach(), masked_freqs)
                r2 = apply_rope(Qs, m.detach() + eps, masked_freqs)
                drot = (r2 - r1) / eps
                residuals = (Qs_rot - targets)
                per_bin_grads.append((2 * residuals * drot).mean().item())

        loss_history.append(loss.item())
        m_history.append(m.item())
        per_bin_grad_history.append(per_bin_grads)

        with torch.no_grad():
            m -= lr * m.grad
        m.grad.zero_()

    print(f"[Isolated | {label}] Final m = {m.item():.4f}, loss = {loss_history[-1]:.6f}")
    return loss_history, m_history, torch.tensor(per_bin_grad_history)

# ============================================================
# Training: Run 2 — Gradient-masked (full target, gradient only from active bins)
# ============================================================
def train_grad_masked(active_bins, label):
    m = torch.tensor(m_init, requires_grad=True)

    with torch.no_grad():
        # Full target — all bins
        targets = apply_rope(Qs, torch.tensor(m_star), freqs)

    loss_history, m_history, per_bin_grad_history = [], [], []

    for step in range(steps):
        # Full forward, but only active_bins backprop to m
        Qs_rot = apply_rope_grad_masked(Qs, m, freqs, active_bins)
        loss = ((Qs_rot - targets) ** 2).mean()
        loss.backward()

        # Per-bin gradient decomposition
        with torch.no_grad():
            per_bin_grads = []
            for i in range(len(freqs)):
                if i not in active_bins:
                    per_bin_grads.append(0.0)
                    continue
                masked_freqs = torch.zeros_like(freqs)
                masked_freqs[i] = freqs[i]
                eps = 1e-5
                r1 = apply_rope(Qs, m.detach(), masked_freqs)
                r2 = apply_rope(Qs, m.detach() + eps, masked_freqs)
                drot = (r2 - r1) / eps
                # Residual is from FULL forward
                residuals = (Qs_rot - targets)
                per_bin_grads.append((2 * residuals * drot).mean().item())

        loss_history.append(loss.item())
        m_history.append(m.item())
        per_bin_grad_history.append(per_bin_grads)

        with torch.no_grad():
            m -= lr * m.grad
        m.grad.zero_()

    print(f"[Masked  | {label}] Final m = {m.item():.4f}, loss = {loss_history[-1]:.6f}")
    return loss_history, m_history, torch.tensor(per_bin_grad_history)

# ============================================================
# Run everything
# ============================================================
results_isolated = {}
results_masked = {}

for name, bins in GROUPS.items():
    print(f"\n--- {name} ---")
    results_isolated[name] = train_isolated(bins, name)
    results_masked[name]   = train_grad_masked(bins, name)

# ============================================================
# Figure 2: Isolated runs — training dynamics
# ============================================================
bin_colors = plt.cm.viridis(torch.linspace(0, 1, len(freqs)).numpy())

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(f"Isolated runs (each group has own target) | m*={m_star}", fontsize=13)

ax = axes2[0, 0]
for name in GROUPS:
    _, m_hist, _ = results_isolated[name]
    ax.plot(m_hist, color=group_colors[name], linewidth=1.2, label=name)
ax.axhline(m_star, color="green", linestyle="--", linewidth=1, label=f"m* = {m_star}")
ax.set_ylabel("m"); ax.set_xlabel("Step"); ax.set_title("m trajectory"); ax.legend(fontsize=7)

ax = axes2[0, 1]
for name in GROUPS:
    loss_hist, _, _ = results_isolated[name]
    ax.semilogy(loss_hist, color=group_colors[name], linewidth=1.2, label=name)
ax.set_ylabel("Loss"); ax.set_xlabel("Step"); ax.set_title("Training loss"); ax.legend(fontsize=7)

ax = axes2[1, 0]
_, _, bin_hist = results_isolated["All bins (0-7)"]
for i in range(len(freqs)):
    ax.plot(bin_hist[:, i].numpy(), color=bin_colors[i], linewidth=0.8, label=f"bin {i} (θ={freqs[i].item():.4f})")
ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
ax.set_ylabel("Grad contrib"); ax.set_xlabel("Step")
ax.set_title("Per-bin grads — All bins run"); ax.legend(fontsize=6, ncol=2, loc="upper right")

ax = axes2[1, 1]
for name in ["High freq (bins 0-2)", "Mid freq  (bins 3-4)", "Low freq  (bins 5-7)"]:
    _, _, bin_hist = results_isolated[name]
    for i in GROUPS[name]:
        ax.plot(bin_hist[:, i].numpy(), color=bin_colors[i], linewidth=0.8,
                label=f"bin {i} (θ={freqs[i].item():.4f})")
ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
ax.set_ylabel("Grad contrib"); ax.set_xlabel("Step")
ax.set_title("Per-bin grads — Isolated group runs"); ax.legend(fontsize=6, ncol=2, loc="upper right")

plt.tight_layout()
plt.savefig("rope_isolated_dynamics.png", dpi=150)
plt.show()

# ============================================================
# Figure 3: Gradient-masked runs — training dynamics
# ============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle(f"Gradient-masked runs (full target, grad only from subset) | m*={m_star}", fontsize=13)

ax = axes3[0, 0]
for name in GROUPS:
    _, m_hist, _ = results_masked[name]
    ax.plot(m_hist, color=group_colors[name], linewidth=1.2, label=name)
ax.axhline(m_star, color="green", linestyle="--", linewidth=1, label=f"m* = {m_star}")
ax.set_ylabel("m"); ax.set_xlabel("Step"); ax.set_title("m trajectory"); ax.legend(fontsize=7)

ax = axes3[0, 1]
for name in GROUPS:
    loss_hist, _, _ = results_masked[name]
    ax.semilogy(loss_hist, color=group_colors[name], linewidth=1.2, label=name)
ax.set_ylabel("Loss"); ax.set_xlabel("Step"); ax.set_title("Training loss"); ax.legend(fontsize=7)

ax = axes3[1, 0]
_, _, bin_hist = results_masked["All bins (0-7)"]
for i in range(len(freqs)):
    ax.plot(bin_hist[:, i].numpy(), color=bin_colors[i], linewidth=0.8, label=f"bin {i} (θ={freqs[i].item():.4f})")
ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
ax.set_ylabel("Grad contrib"); ax.set_xlabel("Step")
ax.set_title("Per-bin grads — All bins run"); ax.legend(fontsize=6, ncol=2, loc="upper right")

ax = axes3[1, 1]
for name in ["High freq (bins 0-2)", "Mid freq  (bins 3-4)", "Low freq  (bins 5-7)"]:
    _, _, bin_hist = results_masked[name]
    for i in GROUPS[name]:
        ax.plot(bin_hist[:, i].numpy(), color=bin_colors[i], linewidth=0.8,
                label=f"bin {i} (θ={freqs[i].item():.4f})")
ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
ax.set_ylabel("Grad contrib"); ax.set_xlabel("Step")
ax.set_title("Per-bin grads — Masked group runs"); ax.legend(fontsize=6, ncol=2, loc="upper right")

plt.tight_layout()
plt.savefig("rope_masked_dynamics.png", dpi=150)
plt.show()

print("\nDone. Check:")
print("  rope_loss_landscape.png")
print("  rope_isolated_dynamics.png")
print("  rope_masked_dynamics.png")