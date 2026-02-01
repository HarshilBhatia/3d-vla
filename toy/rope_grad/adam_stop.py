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
# Stopgrad schedules: stop_top_k(t) → continuous bin weights
# bin 0 = highest freq, bin 7 = lowest freq
# stop_top_k = 7 → only bin 7 active
# stop_top_k = 0 → all bins active
# ============================================================
def get_bin_weights(stop_top_k):
    weights = torch.ones(num_bins)
    k_floor = int(math.floor(stop_top_k))
    k_frac = stop_top_k - k_floor
    for i in range(num_bins):
        if i < k_floor:
            weights[i] = 0.0
        elif i == k_floor:
            weights[i] = 1.0 - k_frac
    return weights

def schedule_none(t, steps):
    return 0.0

def schedule_linear(t, steps):
    return (num_bins - 1) * (1.0 - t / (steps - 1))

def schedule_cosine(t, steps):
    return (num_bins - 1) * 0.5 * (1.0 + math.cos(math.pi * t / (steps - 1)))

# ============================================================
# Runs
# ============================================================
RUNS = {
    # "SGD (no schedule)":          {"schedule": schedule_none,   "use_adam": False},
    # "SGD + Linear schedule":      {"schedule": schedule_linear, "use_adam": False},
    # "SGD + Cosine schedule":      {"schedule": schedule_cosine, "use_adam": False},
    "Adam + no schedule":         {"schedule": schedule_none,   "use_adam": True},
    "Adam + Linear schedule":     {"schedule": schedule_linear, "use_adam": True},
    "Adam + Cosine schedule":     {"schedule": schedule_cosine, "use_adam": True},
}

run_colors = {
    # "SGD (no schedule)":          "steelblue",
    # "SGD + Linear schedule":      "dodgerblue",
    # "SGD + Cosine schedule":      "navy",
    "Adam + no schedule":         "crimson",
    "Adam + Linear schedule":     "darkorange",
    "Adam + Cosine schedule":     "magenta",
}

def train(schedule_fn, use_adam, label):
    m = m_init

    # Per-bin Adam state
    adam_m1 = torch.zeros(num_bins)
    adam_m2 = torch.zeros(num_bins)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    logs = {"loss": [], "m": [], "stop_top_k": [], "bin_weights": [], "final_contribs": []}

    for t in range(steps):
        loss, per_bin_grads = get_per_bin_grads(torch.tensor(m))
        g = torch.tensor(per_bin_grads)

        # Stopgrad: get continuous bin weights from schedule
        k = schedule_fn(t, steps)
        weights = get_bin_weights(k)

        # Apply stopgrad weights to raw grads
        g_masked = g * weights

        if use_adam:
            # Per-bin Adam on the masked gradients
            adam_m1 = beta1 * adam_m1 + (1 - beta1) * g_masked
            adam_m2 = beta2 * adam_m2 + (1 - beta2) * g_masked ** 2

            step_t = t + 1
            m1_hat = adam_m1 / (1 - beta1 ** step_t)
            m2_hat = adam_m2 / (1 - beta2 ** step_t)

            contribs = m1_hat / (m2_hat.sqrt() + adam_eps)
            update = contribs.sum().item()
        else:
            # Plain SGD: just sum masked grads
            contribs = g_masked
            update = contribs.sum().item()

        m = m - lr * update

        logs["loss"].append(loss)
        logs["m"].append(m)
        logs["stop_top_k"].append(k)
        logs["bin_weights"].append(weights.tolist())
        logs["final_contribs"].append(contribs.tolist())

    print(f"[{label}] Final m = {m:.4f}, loss = {logs['loss'][-1]:.6f}")
    return logs

# ============================================================
# Run everything
# ============================================================
results = {}
for name, cfg in RUNS.items():
    print(f"Running {name}...")
    results[name] = train(cfg["schedule"], cfg["use_adam"], name)

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

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle(f"Adam with Stopgrad Schedules\nm* = {m_star}, m_init = {m_init}, lr = {lr}, steps = {steps}", 
             fontsize=14, fontweight='bold', y=0.995)

# --- Row 0: m trajectory ---
ax = axes[0]
for name in RUNS:
    ax.plot(results[name]["m"], color=run_colors[name], linewidth=2.5, alpha=0.9, label=name)
ax.axhline(m_star, color="green", linestyle="--", linewidth=2, label=f"Target: m* = {m_star}", alpha=0.8)
ax.set_ylabel("Position Parameter (m)", fontweight='bold')
ax.set_title("Position Parameter Trajectory", fontweight='bold', pad=10)
ax.legend(fontsize=9, loc="best", framealpha=0.95, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([m_init - 5, m_star + 5])

# --- Row 1: Loss ---
ax = axes[1]
for name in RUNS:
    ax.semilogy(results[name]["loss"], color=run_colors[name], linewidth=2.5, alpha=0.9, label=name)
ax.set_ylabel("Loss (log scale)", fontweight='bold')
ax.set_xlabel("Training Step", fontweight='bold')
ax.set_title("Training Loss (MSE)", fontweight='bold', pad=10)
ax.legend(fontsize=9, loc="best", framealpha=0.95, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

plt.tight_layout(rect=[0, 0, 1, 0.99])
filename = "rope_adam_stopgrad.png"
plt.savefig(filename, dpi=200, bbox_inches='tight')
plt.show()
print(f"Saved main plot: {filename}")

# ============================================================
# Per-Bin Contributions Plot (3 Adam schedules)
# ============================================================
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
fig2.suptitle(f"Per-Bin Adam Contributions with Different Stopgrad Schedules\nm* = {m_star}, m_init = {m_init}, lr = {lr}, steps = {steps}", 
              fontsize=14, fontweight='bold', y=0.995)

schedule_order = ["Adam + no schedule", "Adam + Linear schedule", "Adam + Cosine schedule"]

for idx, name in enumerate(schedule_order):
    ax = axes2[idx]
    contribs_tensor = torch.tensor(results[name]["final_contribs"])  # [steps, num_bins]
    
    # Plot each bin's contribution over time
    for i in range(num_bins):
        ax.plot(contribs_tensor[:, i].numpy(), 
                color=bin_colors[i], 
                linewidth=1.5, 
                alpha=0.8,
                label=f"bin {i} (f={freqs[i].item():.4f})")
    
    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
    ax.set_ylabel("Adam Contribution", fontweight='bold')
    ax.set_title(f"{name} — Per-Bin Adam Contributions", fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Only show legend for first plot to save space
    if idx == 0:
        ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.95, edgecolor='gray')

axes2[-1].set_xlabel("Training Step", fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.99])
filename2 = "rope_adam_stopgrad_per_bin.png"
plt.savefig(filename2, dpi=200, bbox_inches='tight')
plt.show()
print(f"Saved per-bin contributions plot: {filename2}")
print("\nDone!")