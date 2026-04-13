import torch
import math
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
lr = 10
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
    with torch.no_grad():
        Qs_rot = apply_rope(Qs, m_val, freqs)
        loss = ((Qs_rot - targets) ** 2).mean()
        residuals = Qs_rot - targets

        per_bin_grads = []
        eps = 1e-5
        for i in range(num_bins):
            mf = torch.zeros_like(freqs)
            mf[i] = freqs[i]
            r_plus = apply_rope(Qs, m_val + eps, mf)
            r_minus = apply_rope(Qs, m_val, mf)
            drot = (r_plus - r_minus) / eps
            per_bin_grads.append((2 * residuals * drot).mean().item())

    return loss.item(), per_bin_grads

# ============================================================
# Schedules: stop_top_k(t) as a function of step
#
# stop_top_k is a float in [0, num_bins-1].
# At any step, per-bin weight is:
#   bin i < floor(stop_top_k)  → weight 0  (fully blocked)
#   bin i == floor(stop_top_k) → weight = frac(stop_top_k) inverted
#                                 i.e. 1 - (stop_top_k - floor)
#                                 so as stop_top_k decreases past i, weight ramps 0→1
#   bin i > floor(stop_top_k)  → weight 1  (fully active)
# ============================================================
def get_bin_weights(stop_top_k):
    """Returns per-bin weights [0..1] given a float stop_top_k."""
    weights = torch.ones(num_bins)
    k_floor = min(int(math.floor(stop_top_k)), num_bins - 2) # avoid last bin
    k_frac = stop_top_k - k_floor

    for i in range(num_bins):
        if i < k_floor:
            weights[i] = 0.0          # fully blocked
        elif i == k_floor:
            weights[i] = 1.0 - k_frac # partially active (ramps 0→1 as k crosses this bin)
        # else: stays 1.0
    return weights

def schedule_none(t, steps):
    """No stopgrad — all bins active always."""
    return 0.0

def schedule_linear(t, steps):
    """Linear decay from (num_bins-1) → 0 over training."""
    return (num_bins - 1) * (1.0 - t / (steps - 1))

def schedule_cosine(t, steps):
    """Cosine decay from (num_bins-1) → 0. Slower start, faster mid, slower end."""
    return (num_bins - 1) * 0.5 * (1.0 + math.cos(math.pi * t / (steps - 1)))

SCHEDULES = {
    "No schedule (SGD)": schedule_none,
    "Linear":            schedule_linear,
    "Cosine":            schedule_cosine,
}

schedule_colors = {
    "No schedule (SGD)": "steelblue",
    "Linear":            "darkorange",
    "Cosine":            "magenta",
}

# ============================================================
# Training loop
# ============================================================
def run(schedule_fn, label):
    m = m_init
    logs = {"loss": [], "m": [], "stop_top_k": [], "bin_weights": [], "weighted_grads": []}

    for t in range(steps):
        loss, per_bin_grads = get_per_bin_grads(torch.tensor(m))

        # Get current schedule value and bin weights
        k = schedule_fn(t, steps)
        weights = get_bin_weights(k)

        # Apply weights to per-bin grads
        g = torch.tensor(per_bin_grads)
        weighted_g = (g * weights)
        update = weighted_g.sum().item()

        m = m - lr * update

        logs["loss"].append(loss)
        logs["m"].append(m)
        logs["stop_top_k"].append(k)
        logs["bin_weights"].append(weights.tolist())
        logs["weighted_grads"].append(weighted_g.tolist())

    print(f"[{label}] Final m = {m:.4f}, loss = {logs['loss'][-1]:.6f}")
    return logs

# ============================================================
# Run all schedules
# ============================================================
results = {}
for name, fn in SCHEDULES.items():
    print(f"Running {name}...")
    results[name] = run(fn, name)

# ============================================================
# Plot
# ============================================================
# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

bin_colors = plt.cm.viridis(torch.linspace(0, 1, num_bins).numpy())

fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
fig.suptitle(f"RoPE Stopgrad Schedules: Continuous Nested Unblocking\nm* = {m_star}, m_init = {m_init}, lr = {lr}, steps = {steps}", 
             fontsize=14, fontweight='bold', y=0.995)

# --- Row 0: m trajectory ---
ax = axes[0]
for name in SCHEDULES:
    ax.plot(results[name]["m"], color=schedule_colors[name], linewidth=2.5, label=name, alpha=0.9)
ax.axhline(m_star, color="crimson", linestyle="--", linewidth=2, label=f"Target: m* = {m_star}", alpha=0.8)
ax.set_ylabel("Position Parameter (m)", fontweight='bold')
ax.set_title("Position Parameter Trajectory", fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([m_init - 5, m_star + 5])

# --- Row 1: Loss ---
ax = axes[1]
for name in SCHEDULES:
    ax.semilogy(results[name]["loss"], color=schedule_colors[name], linewidth=2.5, label=name, alpha=0.9)
ax.set_ylabel("Loss (log scale)", fontweight='bold')
ax.set_title("Training Loss (MSE)", fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

# --- Row 2: stop_top_k over time (the schedule itself) ---
ax = axes[2]
for name in SCHEDULES:
    ax.plot(results[name]["stop_top_k"], color=schedule_colors[name], linewidth=2.5, label=name, alpha=0.9)
ax.set_ylabel("Stop Top-K Value", fontweight='bold')
ax.set_title("Stopgrad Schedule Over Training", fontweight='bold', pad=10)
ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
# Mark bin boundaries with labels
for i in range(num_bins):
    ax.axhline(i, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    if i % 2 == 0:  # Label every other bin to avoid clutter
        ax.text(steps * 1.01, i, f'bin {i}', fontsize=8, va='center', color='gray', alpha=0.7)
ax.set_ylim([-0.5, num_bins - 0.5])


plt.tight_layout(rect=[0, 0, 1, 0.99])
filename = "no_last_bin_rope_stopgrad_schedule.png"
plt.savefig(filename, dpi=200, bbox_inches='tight')
plt.show()
print(f"\nDone. Check {filename}")