import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ============================================================
# Config
# ============================================================
dim = 16              # 8 frequency bins
base = 10000
num_pairs = 32
m_star = 50.0
m_init = 0.0
lr = 0.1
steps = 2000

# ============================================================
# Setup
# ============================================================
Qs = torch.randn(num_pairs, dim)

def get_freqs(dim, base):
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

freqs = get_freqs(dim, base)
num_bins = len(freqs)

def apply_rope(x, m, freqs):
    angles = m * freqs
    cos_a, sin_a = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.zeros_like(x)
    out[..., 0::2] = x1 * cos_a - x2 * sin_a
    out[..., 1::2] = x1 * sin_a + x2 * cos_a
    return out

with torch.no_grad():
    targets = apply_rope(Qs, torch.tensor(m_star), freqs)

print("Freqs:", [f"{f:.4f}" for f in freqs.tolist()])
print(f"Rotations over [0, m*]: {[f'{m_star*f/(2*3.14159):.2f}' for f in freqs.tolist()]}\n")

# ============================================================
# Per-bin gradient via finite diff
# ============================================================
def get_per_bin_grads(m_val):
    """Returns: loss, full_grad, per_bin_grads (list of scalars)"""
    with torch.no_grad():
        Qs_rot = apply_rope(Qs, m_val, freqs)
        loss = ((Qs_rot - targets) ** 2).mean()
        residuals = Qs_rot - targets  # [N, dim]

        per_bin_grads = []
        eps = 1e-5
        for i in range(num_bins):
            mf = torch.zeros_like(freqs)
            mf[i] = freqs[i]
            r_plus = apply_rope(Qs, m_val + eps, mf)
            r_minus = apply_rope(Qs, m_val, mf)
            drot = (r_plus - r_minus) / eps          # [N, dim]
            bin_grad = (2 * residuals * drot).mean().item()
            per_bin_grads.append(bin_grad)

    full_grad = sum(per_bin_grads)
    return loss.item(), full_grad, per_bin_grads

# ============================================================
# Run 1: Plain SGD — sum raw per-bin grads, single step
# ============================================================
def run_sgd():
    m = m_init
    logs = {"loss": [], "m": [], "raw_grads": []}

    for step in range(steps):
        loss, full_grad, per_bin_grads = get_per_bin_grads(torch.tensor(m))
        m = m - lr * full_grad

        logs["loss"].append(loss)
        logs["m"].append(m)
        logs["raw_grads"].append(per_bin_grads)

    print(f"[SGD] Final m = {m:.4f}, loss = {logs['loss'][-1]:.6f}")
    return logs

# ============================================================
# Run 2: Per-bin Adam — normalize each bin's grad before summing
# ============================================================
def run_perbin_adam():
    m = m_init
    m1 = torch.zeros(num_bins)
    m2 = torch.zeros(num_bins)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    logs = {"loss": [], "m": [], "raw_grads": [], "adam_contribs": []}

    for step in range(steps):
        loss, full_grad, per_bin_grads = get_per_bin_grads(torch.tensor(m))
        g = torch.tensor(per_bin_grads)

        # Adam moment updates per bin
        m1 = beta1 * m1 + (1 - beta1) * g
        m2 = beta2 * m2 + (1 - beta2) * g ** 2

        # Bias correction
        t = step + 1
        m1_hat = m1 / (1 - beta1 ** t)
        m2_hat = m2 / (1 - beta2 ** t)

        # Normalized contribution per bin, then sum
        adam_contribs = m1_hat / (m2_hat.sqrt() + eps)
        update = adam_contribs.sum().item()
        m = m - lr * update

        logs["loss"].append(loss)
        logs["m"].append(m)
        logs["raw_grads"].append(per_bin_grads)
        logs["adam_contribs"].append(adam_contribs.tolist())

    print(f"[Per-bin Adam] Final m = {m:.4f}, loss = {logs['loss'][-1]:.6f}")
    return logs

# ============================================================
# Run
# ============================================================
print("Running SGD...")
sgd_logs = run_sgd()
print("Running per-bin Adam...")
adam_logs = run_perbin_adam()

# ============================================================
# Plot
# ============================================================
# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 14

sgd_raw  = torch.tensor(sgd_logs["raw_grads"])       # [steps, bins]
adam_raw = torch.tensor(adam_logs["raw_grads"])       # [steps, bins]
adam_norm = torch.tensor(adam_logs["adam_contribs"])  # [steps, bins]

bin_colors = plt.cm.viridis(torch.linspace(0, 1, num_bins).numpy())

fig, axes = plt.subplots(3, 2, figsize=(16, 13))
fig.suptitle(f"SGD vs Per-Bin Adam on RoPE Optimization\nm* = {m_star}, m_init = {m_init}, lr = {lr}, dim = {dim}, base = {base}, steps = {steps}", 
             fontsize=14, fontweight='bold', y=0.995)

# --- Col 0: SGD ---
ax = axes[0, 0]
ax.plot(sgd_logs["m"], color="#2E86AB", linewidth=2.5, alpha=0.9, label="SGD trajectory")
ax.axhline(m_star, color="crimson", linestyle="--", linewidth=2, label=f"Target: m* = {m_star}", alpha=0.8)
ax.set_ylabel("Position Parameter (m)", fontweight='bold')
ax.set_title("SGD — Position Trajectory", fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([m_init - 5, m_star + 5])

ax = axes[1, 0]
ax.semilogy(sgd_logs["loss"], color="#2E86AB", linewidth=2.5, alpha=0.9)
ax.set_ylabel("Loss (log scale)", fontweight='bold')
ax.set_title("SGD — Training Loss (MSE)", fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

ax = axes[2, 0]
for i in range(num_bins):
    ax.plot(sgd_raw[:, i].numpy(), color=bin_colors[i], linewidth=1.5, alpha=0.8,
            label=f"bin {i} (f={freqs[i].item():.4f})")
ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
ax.set_ylabel("Gradient Contribution", fontweight='bold')
ax.set_xlabel("Training Step", fontweight='bold')
ax.set_title("SGD — Raw Per-Bin Gradients", fontweight='bold', pad=10)
ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.95, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# --- Col 1: Per-bin Adam ---
ax = axes[0, 1]
ax.plot(adam_logs["m"], color="#A23B72", linewidth=2.5, alpha=0.9, label="Adam trajectory")
ax.axhline(m_star, color="crimson", linestyle="--", linewidth=2, label=f"Target: m* = {m_star}", alpha=0.8)
ax.set_ylabel("Position Parameter (m)", fontweight='bold')
ax.set_title("Per-Bin Adam — Position Trajectory", fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([m_init - 5, m_star + 5])

ax = axes[1, 1]
ax.semilogy(adam_logs["loss"], color="#A23B72", linewidth=2.5, alpha=0.9)
ax.set_ylabel("Loss (log scale)", fontweight='bold')
ax.set_title("Per-Bin Adam — Training Loss (MSE)", fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

ax = axes[2, 1]
for i in range(num_bins):
    ax.plot(adam_norm[:, i].numpy(), color=bin_colors[i], linewidth=1.5, alpha=0.8,
            label=f"bin {i} (f={freqs[i].item():.4f})")
ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
ax.set_ylabel("Adam-Normalized Contribution", fontweight='bold')
ax.set_xlabel("Training Step", fontweight='bold')
ax.set_title("Per-Bin Adam — Normalized Contributions", fontweight='bold', pad=10)
ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.95, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.99])
filename = "rope_adam_vs_sgd.png"
plt.savefig(filename, dpi=200, bbox_inches='tight')
plt.show()
print(f"Saved comparison plot: {filename}")

# ============================================================
# Additional comparison plot: Direct overlay
# ============================================================
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
fig2.suptitle(f"Direct Comparison: SGD vs Per-Bin Adam\nm* = {m_star}, m_init = {m_init}, lr = {lr}", 
              fontsize=14, fontweight='bold')

# Position trajectory comparison
ax = axes2[0]
ax.plot(sgd_logs["m"], color="#2E86AB", linewidth=2.5, alpha=0.9, label="SGD", linestyle='-')
ax.plot(adam_logs["m"], color="#A23B72", linewidth=2.5, alpha=0.9, label="Per-Bin Adam", linestyle='-')
ax.axhline(m_star, color="crimson", linestyle="--", linewidth=2, label=f"Target: m* = {m_star}", alpha=0.8)
ax.set_ylabel("Position Parameter (m)", fontweight='bold', fontsize=12)
ax.set_title("Position Trajectory Comparison", fontweight='bold', pad=10, fontsize=13)
ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([m_init - 5, m_star + 5])

# Loss comparison
ax = axes2[1]
ax.semilogy(sgd_logs["loss"], color="#2E86AB", linewidth=2.5, alpha=0.9, label="SGD")
ax.semilogy(adam_logs["loss"], color="#A23B72", linewidth=2.5, alpha=0.9, label="Per-Bin Adam")
ax.set_ylabel("Loss (log scale)", fontweight='bold', fontsize=12)
ax.set_xlabel("Training Step", fontweight='bold', fontsize=12)
ax.set_title("Training Loss Comparison (MSE)", fontweight='bold', pad=10, fontsize=13)
ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

# Add text annotations for final values
sgd_final_loss = sgd_logs["loss"][-1]
adam_final_loss = adam_logs["loss"][-1]
sgd_final_m = sgd_logs["m"][-1]
adam_final_m = adam_logs["m"][-1]

textstr = f'Final Results:\n'
textstr += f'SGD:   m = {sgd_final_m:.4f}, loss = {sgd_final_loss:.6f}\n'
textstr += f'Adam: m = {adam_final_m:.4f}, loss = {adam_final_loss:.6f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray', linewidth=1.5)
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

plt.tight_layout()
filename2 = "rope_adam_vs_sgd_comparison.png"
plt.savefig(filename2, dpi=200, bbox_inches='tight')
plt.show()
print(f"Saved direct comparison plot: {filename2}")
print("\nDone!")