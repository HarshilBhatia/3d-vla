#!/bin/bash
#SBATCH --job-name=debug_delta_m
#SBATCH --output=logs/debug_delta_m_%j.log
#SBATCH --error=logs/debug_delta_m_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=bgkz-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --gres=gpu:1

source ~/.bashrc
micromamba activate gr00t

mkdir -p /u/hbhatia1/3d-vla/logs

cd /u/hbhatia1/3d-vla

echo "=== deltaM GPU smoke test on $(hostname) ==="

python - << 'PYEOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}")

from gr00t.model.modules.dit import AlternateVLDiT

B, T, D, S = 2, 17, 1536, 132  # S=132 = 2 cameras * 66 tokens
OUT_DIM = 14

print("Creating model with use_delta_m=True...")
model = AlternateVLDiT(
    num_layers=4, num_attention_heads=8, attention_head_dim=192,
    output_dim=14, interleave_self_attention=True,
    cross_attention_dim=2048, use_delta_m=True, num_cameras=2,
).cuda().bfloat16()

hidden     = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)
enc        = torch.randn(B, S, 2048, device='cuda', dtype=torch.bfloat16)
img_mask   = torch.zeros(B, S, dtype=torch.bool, device='cuda')
img_mask[:] = True   # all S tokens are image tokens
bbone_mask = torch.ones(B, S, dtype=torch.bool, device='cuda')
positions  = torch.randn(B, S, 3, device='cuda', dtype=torch.bfloat16)
cam_pos    = torch.randn(B, 2, 3, device='cuda', dtype=torch.bfloat16)
timestep   = torch.randint(0, 1000, (B,), device='cuda')

print("Running forward pass with camera_positions_3d...")
out = model(hidden, enc, timestep=timestep,
            encoder_attention_mask=bbone_mask, image_mask=img_mask,
            backbone_attention_mask=bbone_mask, token_positions_3d=positions,
            camera_positions_3d=cam_pos)
print(f"  output shape: {out.shape}")
assert out.shape == (B, T, OUT_DIM), f"Expected {(B, T, OUT_DIM)}, got {out.shape}"
print("  PASS")

print("Running forward pass with camera_positions_3d=None...")
out2 = model(hidden, enc, timestep=timestep,
             encoder_attention_mask=bbone_mask, image_mask=img_mask,
             backbone_attention_mask=bbone_mask, token_positions_3d=positions,
             camera_positions_3d=None)
assert out2.shape == (B, T, OUT_DIM)
print("  PASS")

print("Running backward pass...")
loss = out.mean()
loss.backward()
print("  PASS (backward)")

print("Testing use_delta_m=False (baseline)...")
model2 = AlternateVLDiT(
    num_layers=4, num_attention_heads=8, attention_head_dim=192,
    output_dim=14, interleave_self_attention=True,
    cross_attention_dim=2048, use_delta_m=False,
).cuda().bfloat16()
out3 = model2(hidden.detach(), enc.detach(), timestep=timestep,
              encoder_attention_mask=bbone_mask, image_mask=img_mask,
              backbone_attention_mask=bbone_mask, token_positions_3d=positions)
assert out3.shape == (B, T, OUT_DIM)
print("  PASS")

print("\nALL TESTS PASSED")
PYEOF
