# Camera-frame point clouds + delta_M

**Status:** active
**Week introduced:** [22–29 Sept](../../weekly_logs/2026-09-29.md) (TODO 6)

## Question

Every camera's point cloud (orbital and wrist) stays in **that camera's own
frame**, and extrinsics are never used. Proprio and actions stay in the robot
frame. delta_M (`rope_6d`, layer-wise, all 4 cameras) has to bridge the two
frames.

No miscalibration is applied, since extrinsics aren't used. delta_M is *not*
added to the encoder's proprio attention (decided).

## Setup

| | |
|---|---|
| Config | `config/experiment/peract2_orbital_camframe_rope6d.yaml` |
| ARM | `camframe_rope6d` |
| Training | from scratch, 200k steps, batch 128 |
| Jobs | 3966717 (4 nodes × 1 GPU on `all`, `scripts/multinode/config/camframe_rope6d.yaml`); 3966757 → 3966758 after a crash |

## Results

SBRS **53.3 at 200k**. Coarse tasks work; fine tasks sit around 20.

## Reading

If it works on G7 but falls off in the
[viewpoint sweep](g7-viewpoint-sweep.md), the training camera distribution is
too small, and the follow-up is regenerating data with densely sampled cameras.

## Caveats and open questions

- One 6×6 delta_M is shared across all positional-encoding frequencies and can
  express at most about 121° of rotation. It probably cannot represent a full
  camera-to-robot transform.
- The proprio attention stays geometrically inconsistent.
- A validation DataLoader worker segfaulted at step 68k (job 3966756;
  continuation 3966757 resumed there). If it recurs at a validation pass
  (every 4k steps), it is systematic.
- The automatic SBRS ladder runs two calibration conditions that give this
  model identical inputs, so that eval costs double. **Open: trim the ladder to
  a single condition?**
