# 3D RoPE Diagnostic Analysis

**Date:** 2026-03-21

## Problem
Training loss curves for 3D RoPE runs are identical to the baseline (no 3D RoPE), despite:
- Depth shards loading correctly (357 shards, 125/132 image tokens non-zero per sample)
- Correct cam2cam extrinsics being used (confirmed via epipolar + depth consistency tests)
- 3D positions flowing correctly through the data pipeline

## Setup
- Model: GR00T-N1.6-3B, DiT-only tuning
- Embodiment: OXE_DROID_EXT2 (ext1 + ext2 cameras)
- Backbone cache: `/work/nvme/bgkz/droid_multilab_cache_ext2/` (357 shards, 365K samples)
- Depth cache: `/work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2/`
- 3D RoPE: applied to image KEY vectors in DiT cross-attention blocks (2, 6, 10, ...)
- Runs compared: `multilab-3d-cam2cam-ext2` vs `multilab-baseline-ext2`

---

## Diagnostic Steps

### Step 1 — Intra-sample position spread (`analyze_rope_angles.py`)
**Hypothesis:** Token positions within a sample have too little spatial spread → RoPE cannot discriminate between tokens.

**Result: PASS**
- Intra-sample XYZ std across image tokens: X=0.886m, Y=0.554m, Z=0.739m
- Max rotation angles: X=199.6°, Y=130.7°, Z=203.4°
- All 3 axes exceed the 0.3m threshold

**Conclusion:** Positions are spatially discriminative. RoPE *should* be able to distinguish tokens.

---

### Step 6 — Cross-sample EEF and token position variance (`analyze_position_variance.py`)
**Hypothesis:** EEF/token positions have near-zero variance across training set → same RoPE encoding for every sample → no gradient signal.

**Result: MARGINAL**
- EEF position std: ~0.116–0.168m per axis (mean 0.145m, below 0.2m threshold)
- EEF position range: ~0.7m across samples (meaningful)
- Token mean position std: 0.21–0.28m across samples
- EEF ↔ token position correlation: ~0 (positions don't trivially correlate with EEF)

**Conclusion:** Enough variance exists to provide a meaningful positional signal. Not the root cause.

---

### Step 2 — Ablate image cross-attention (`ablate_image_attention.py`)
**Hypothesis:** The DiT relies primarily on text cross-attention blocks and barely uses image cross-attention. RoPE on image blocks would then have no effect.

**Test:** Load 3D checkpoint. Run same batch twice — once normally, once with `image_mask` zeroed (image cross-attention sees nothing). Compare loss.

**Result: FAIL**
```
Batch 1: loss_normal=1.1459  loss_no_image=1.1316  delta=-0.0143  rel_increase=-1.2%
Batch 2: loss_normal=1.0985  loss_no_image=1.1179  delta=+0.0194  rel_increase=+1.8%
Batch 3: loss_normal=1.1333  loss_no_image=1.0663  delta=-0.0670  rel_increase=-5.9%
Batch 4: loss_normal=1.1116  loss_no_image=1.0754  delta=-0.0361  rel_increase=-3.3%
Batch 5: loss_normal=1.1893  loss_no_image=1.1388  delta=-0.0505  rel_increase=-4.2%
Batch 6: loss_normal=1.1320  loss_no_image=1.1705  delta=+0.0385  rel_increase=+3.4%
Batch 7: loss_normal=1.1372  loss_no_image=1.1289  delta=-0.0083  rel_increase=-0.7%
Batch 8: loss_normal=1.0962  loss_no_image=1.1877  delta=+0.0915  rel_increase=+8.3%

RESULTS: mean loss_normal=1.1305  mean loss_no_image=1.1271
Relative increase: -0.3%
VERDICT: FAIL — model ignores image cross-attention
```

**Conclusion:** The model completely ignores image cross-attention. Ablating it changes loss by -0.3% (noise level). This is the **root cause**.

---

## Root Cause

The DiT architecture alternates:
- Text cross-attention blocks (0, 4, 8, ...) — attend to backbone text/image tokens
- Image cross-attention blocks (2, 6, 10, ...) — attend to backbone image tokens only

The model has learned to rely almost entirely on **text cross-attention blocks** and ignores image cross-attention blocks. Since 3D RoPE is only applied to image key vectors in image cross-attention blocks, it has **zero impact on training loss** — not because the implementation is wrong, but because those blocks are functionally unused.

### Why text blocks dominate
Both text and image cross-attention blocks receive the same backbone features. Text blocks receive all backbone tokens (text + image), while image blocks receive only image tokens. The model likely learned early that text blocks alone provide sufficient signal, making image blocks redundant.

---

## Implication

**3D RoPE cannot work in its current formulation.** Any further tuning — EEF-relative rope, frequency sweeps (`rope_base_freq`), position noise — would still show zero effect because the blocks being modified are not used.

---

## Potential Fixes

### Option 1: Apply 3D RoPE in the backbone (ViT self-attention)
Apply positional RoPE at the ViT level, where image tokens are actually processed. This requires unprojecting 2D patch positions to 3D using depth, then applying RoPE in ViT self-attention. More invasive but targets where visual features are actually computed.

### Option 2: Force image cross-attention to be used
Train with text tokens masked or down-weighted, forcing the model to learn from image cross-attention. Could be done via a curriculum: start with full text, progressively mask. Risk: may degrade language-following.

### Option 3: Architectural change — remove text cross-attention, keep image only
Replace text cross-attention blocks with image cross-attention blocks (or merge them). This forces the model to use image features directly. Requires architectural modification and re-training from scratch.

### Option 4: Shared cross-attention with 3D RoPE
Instead of separate text/image blocks, use a single cross-attention block that attends to all backbone tokens, with 3D RoPE applied only to image token keys. Text tokens get zero position (identity rotation). This preserves language conditioning while injecting spatial structure.
