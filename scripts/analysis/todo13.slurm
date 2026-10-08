#!/bin/bash
#SBATCH --job-name=todo13
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/todo13-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18,grogu-4-23,grogu-1-20
# TODO 13: PART=follow (following coefficients) or patch (activation patching) on the four full-patch models, loaded through load_model_for_evaluation.
cd /home/harshilb/3dfa_unified
L=train_logs/PerAct2
M=(--model fp=$L/peract2_orbital_vid_deltam_fullpatch_finetune_300k_a6000_b64/best.pth:raw:ckpt
   --model fp_d8=$L/peract2_orbital_vid_deltam_fullpatch_d8/best.pth:raw:ckpt
   --model fp_r3d_refine=$L/peract2_orbital_vid_deltam_refine_fullpatch_rope3d/best.pth:raw:ckpt
   --model fp_r3d_direct=$L/peract2_orbital_vid_deltam_direct_fullpatch_rope3d/best.pth:raw:ckpt)
if [ "$PART" = follow ]; then
  /home/harshilb/miniconda3/envs/3dfa/bin/python scripts/analysis/follow_miscal.py --eval-loader "${M[@]}" --out-dir results/analysis/todo13/follow_miscal "$@"
else
  /home/harshilb/miniconda3/envs/3dfa/bin/python scripts/analysis/activation_patching.py --eval-loader "${M[@]}" --out-dir results/analysis/todo13/activation_patching "$@"
fi
