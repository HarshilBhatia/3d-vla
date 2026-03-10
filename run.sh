# export NUM_GPUS=1

# CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
#   --base-model-path nvidia/GR00T-N1.6-3B \
#   --dataset-path /ocean/projects/cis240058p/hbhatia1/droid_100 \
#   --embodiment-tag OXE_DROID \
#   --num-gpus $NUM_GPUS \
#   --output-dir /ocean/projects/cis240058p/hbhatia1/outputs/test_run \
#   --max-steps 10 \
#   --save-steps 5 \
#   --save-total-limit 1 \
#   --global-batch-size 4 \
#   --dataloader-num-workers 2


# data/droid_v2_small/meta/modality.json
python cache_backbone_features.py   --dataset-path /ocean/projects/cis240058p/hbhatia1/data/droid_raw_large_superset   --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_v2_large_superset_cache   --batch-size 128