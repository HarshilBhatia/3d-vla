#!/bin/bash
#SBATCH --job-name=fp_suite
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/fp_suite-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18,grogu-1-20,grogu-4-23
# TODO 12: PART=attn | dm | ctok on the four full-patch models (raw, saved K=5, eval-faithful load); extra args pass through.
cd /home/harshilb/3dfa_unified
PY=/home/harshilb/miniconda3/envs/3dfa/bin/python
L=train_logs/PerAct2
M=(--model fp=$L/peract2_orbital_vid_deltam_fullpatch_finetune_300k_a6000_b64/best.pth:raw:ckpt
   --model fp_d8=$L/peract2_orbital_vid_deltam_fullpatch_d8/best.pth:raw:ckpt
   --model fp_r3d_refine=$L/peract2_orbital_vid_deltam_refine_fullpatch_rope3d/best.pth:raw:ckpt
   --model fp_r3d_direct=$L/peract2_orbital_vid_deltam_direct_fullpatch_rope3d/best.pth:raw:ckpt)
O=${OUT:-results/analysis/todo12}
case "$PART" in
  attn) $PY scripts/analysis/camera_attention.py --faithful "${M[@]}" --out-dir $O/camera_attention "$@" ;;
  dm) $PY scripts/analysis/delta_m_per_layer.py --faithful "${M[@]}" --out-dir $O/delta_m_per_layer "$@" &&
      $PY scripts/analysis/delta_m_trunk_activity.py --faithful "${M[@]}" --out $O/delta_m_per_layer/trunk_activity.json ;;
  ctok) $PY scripts/analysis/ctoken_behaviour.py "${M[@]}" --out-dir $O/ctoken_behaviour "$@" ;;
esac
