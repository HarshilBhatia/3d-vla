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
ema_beta = 0.99       # EMA decay for mean and variance tracking

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
# Run 1: Plain SGD
# ============================================================
def run_sgd():
    m = m_init
    logs = {"loss": [], "m": [], "raw_grads": []}

    for step in range(steps):
        loss, per_bin_grads = get_per_bin_grads(torch.tensor(m))
        update = sum(per_bin_grads)
        m = m - lr * update

        logs["loss"].append(loss)
        logs["m"].append(m)
        logs["raw_grads"].append(per_bin_grads)

    print(f"[SGD] Final m = {m:.4f}, loss = {logs['loss'][-1]:.6f}")
    return logs

# ============================================================
# Run 2: Per-bin SNR weighting
#
# Per bin, track:
#   mean[i] = EMA of g_i          (signal)
#   var[i]  = EMA of (g_i - mean[i])^2  (noise)
#   SNR[i]  = |mean[i]| / sqrt(var[i] + eps)
#
# Weight each bin by its SNR before summing.
# High-freq bins oscillate → mean cancels → SNR ≈ 0 → down-weighted
# Low-freq bins consistent → mean strong, var small → SNR high → up-weighted
# Self-correcting: as m → m*, high-freq stabilizes → SNR rises → unlocks
# ============================================================
def run_snr():
    m = m_init
    mean = torch.zeros(num_bins)
    var  = torch.ones(num_bins)   # init to 1 so SNR starts at 0 (not div by 0)

    logs = {"loss": [], "m": [], "raw_grads": [], "snr": [], "weighted_grads": []}

    for step in range(steps):
        loss, per_bin_grads = get_per_bin_grads(torch.tensor(m))
        g = torch.tensor(per_bin_grads)

        # Update EMA of mean
        mean = ema_beta * mean + (1 - ema_beta) * g

        # Update EMA of variance (around current mean)
        var = ema_beta * var + (1 - ema_beta) * (g - mean) ** 2

        # Compute SNR per bin
        snr = mean.abs() / (var.sqrt() + 1e-8)

        # Weight raw grads by SNR, then sum
        weighted_g = g * snr
        update = weighted_g.sum().item()
        m = m - lr * update

        logs["loss"].append(loss)
        logs["m"].append(m)
        logs["raw_grads"].append(per_bin_grads)
        logs["snr"].append(snr.tolist())
        logs["weighted_grads"].append(weighted_g.tolist())

    print(f"[SNR] Final m = {m:.4f}, loss = {logs['loss'][-1]:.6f}")
    return logs

# ============================================================
# Run
# ============================================================
print("Running SGD...")
sgd_logs = run_sgd()
print("Running SNR...")
snr_logs = run_snr()

# ============================================================
# Plot
# ============================================================
sgd_raw     = torch.tensor(sgd_logs["raw_grads"])       # [steps, bins]
snr_raw     = torch.tensor(snr_logs["raw_grads"])       # [steps, bins]
snr_vals    = torch.tensor(snr_logs["snr"])             # [steps, bins]
snr_weighted = torch.tensor(snr_logs["weighted_grads"]) # [steps, bins]

bin_colors = plt.cm.viridis(torch.linspace(0, 1, num_bins).numpy())

fig, axes = plt.subplots(4, 2, figsize=(14, 14))
fig.suptitle(f"SGD vs Per-bin SNR | m*={m_star}, m_init={m_init}, ema_beta={ema_beta}", fontsize=13)

# --- Col 0: SGD ---
ax = axes[0, 0]
ax.plot(sgd_logs["m"], color="steelblue", linewidth=1.2)
ax.axhline(m_star, color="red", linestyle="--", label=f"m* = {m_star}")
ax.set_ylabel("m"); ax.set_title("SGD — m trajectory"); ax.legend(fontsize=8)

ax = axes[1, 0]
ax.semilogy(sgd_logs["loss"], color="steelblue", linewidth=1.2)
ax.set_ylabel("Loss"); ax.set_title("SGD — loss")

ax = axes[2, 0]
for i in range(num_bins):
    ax.plot(sgd_raw[:, i].numpy(), color=bin_colors[i], linewidth=0.8,
            label=f"bin {i} (θ={freqs[i].item():.4f})")
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_ylabel("Grad contribution"); ax.set_title("SGD — raw per-bin gradients")
ax.legend(fontsize=6, ncol=2, loc="upper right")

# Empty bottom-left (SGD has no SNR)
axes[3, 0].axis("off")

# --- Col 1: SNR ---
ax = axes[0, 1]
ax.plot(snr_logs["m"], color="magenta", linewidth=1.2)
ax.axhline(m_star, color="red", linestyle="--", label=f"m* = {m_star}")
ax.set_ylabel("m"); ax.set_title("SNR — m trajectory"); ax.legend(fontsize=8)

ax = axes[1, 1]
ax.semilogy(snr_logs["loss"], color="magenta", linewidth=1.2)
ax.set_ylabel("Loss"); ax.set_title("SNR — loss")

ax = axes[2, 1]
for i in range(num_bins):
    ax.plot(snr_weighted[:, i].numpy(), color=bin_colors[i], linewidth=0.8,
            label=f"bin {i} (θ={freqs[i].item():.4f})")
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_ylabel("SNR-weighted grad"); ax.set_title("SNR — weighted per-bin gradients")
ax.legend(fontsize=6, ncol=2, loc="upper right")

# SNR values over time — the key diagnostic
ax = axes[3, 1]
for i in range(num_bins):
    ax.plot(snr_vals[:, i].numpy(), color=bin_colors[i], linewidth=0.8,
            label=f"bin {i} (θ={freqs[i].item():.4f})")
ax.set_ylabel("SNR"); ax.set_xlabel("Step"); ax.set_title("SNR — per-bin SNR over training")
ax.legend(fontsize=6, ncol=2, loc="upper right")

plt.tight_layout()
plt.savefig("rope_snr_vs_sgd.png", dpi=150)
plt.show()
print("\nDone. Check rope_snr_vs_sgd.png")