# PerAct2 experiments prepared for Grogu

These are intentionally separate experiments:

1. `peract2_orbital_2d`: a training-time baseline. It trains from scratch on
   the clean PerAct2 orbital zarr with no training miscalibration. It consumes
   RGB from all four cameras and uses the SigLIP2 visual backbone plus fixed additive sinusoidal positional
   embeddings over image tokens. It does not consume point clouds, XYZ camera
   geometry, geometric RoPE, or predicted extrinsics. `num_history=3`,
   `bimanual=true`, and global batch size is 256 (64 per GPU on four GPUs).
2. `eval_peract2_clean3d_miscal_camera_subset.slurm`: two online evaluations
   of an already-trained clean 3D checkpoint. The first applies the selected
   test-time extrinsic perturbation only to orbital indices `[0,1]`; the second
   applies it only to wrist indices `[2,3]`. RGB and depth remain unchanged;
   only the extrinsics used to create the model's PCD are perturbed.

The default test perturbation is the pinned `5deg` + `5cm` random level and can
be changed with `MISCAL_ROT_LEVEL` and `MISCAL_TRANS_LEVEL`. The default camera
group is `G1`; set `SPAWN_CAMERA_GROUP` to repeat another held-out group.

The training script defaults to the extracted egress zarr at
`/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820/datasets/orbital_peract2/zarr/`.
The online harness defaults to `/grogu/datasets/hbhatia/peract2_test`, which
must be created by extracting the separately shipped `peract2_test_seeds.tar.zst`.
Override `TRAIN_DATA_DIR`, `EVAL_DATA_DIR`, and `CHECKPOINT` when your staging
locations differ.

Example:

```bash
sbatch scripts/train/train_peract2_orbital_2d_grogu.slurm
CHECKPOINT=/path/to/clean3d.pth CAMERA_SET=external sbatch scripts/eval/eval_peract2_clean3d_miscal_camera_subset.slurm
CHECKPOINT=/path/to/clean3d.pth CAMERA_SET=wrist sbatch scripts/eval/eval_peract2_clean3d_miscal_camera_subset.slurm
```
