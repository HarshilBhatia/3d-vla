#!/bin/bash

python -m gr00t.experiment.launch_finetune \
    --base-model-path            nvidia/GR00T-N1.6-3B \
    --dataset-path               /work/nvme/bgkz/droid_raw_large_superset \
    --embodiment-tag             OXE_DROID \
    --output-dir                 /work/hdd/bgkz/hbhatia1/outputs_multilab_baseline_v2 \
    --cached-backbone-dir        /work/nvme/bgkz/droid_multilab_cache \
    --allowed-indices-file       /work/nvme/bgkz/droid_multilab_depths/selected_episodes.json \
    --shard-size                 1024 \
    --episode-sampling-rate      0.1 \
    --global-batch-size          64 \
    --max-steps                  100000 \
    --dataloader-num-workers     2 \
    --save-steps                 1000 \
    --save-total-limit           2 \
    --no-tune-llm \
    --no-tune-visual \
    --no-tune-projector \
    --tune-diffusion-model \
    --num-shards-per-epoch 1 \
    --shard-size 64 \
    --max-steps 3 \
    --dataloader-num-workers 0 \
    --global-batch-size 4 \
    --save-steps 9999 \
    --use-3d-rope 
          