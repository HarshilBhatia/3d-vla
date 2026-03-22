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
# python cache_backbone_features.py   --dataset-path /ocean/projects/cis240058p/hbhatia1/data/droid_raw_large_superset   --output-dir /ocean/projects/cis240058p/hbhatia1/data/droid_v2_large_superset_cache   --batch-size 128


# python data_processing/visualise_epipolar.py \
#       --raw-dir /work/nvme/bgkz/droid_multilab_raw \
#       --depth-dir /work/nvme/bgkz/droid_multilab_depths \
#       --canonical-id ILIAD+50aee79f+2023-07-12-21h-13m-44s \
#       --cam2cam-json /work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json \
#       --output epipolar.mp4         

#   python data_processing/visualise_eef_projection.py \
#       --raw-dir /work/nvme/bgkz/droid_multilab_raw \
#       --depth-dir /work/nvme/bgkz/droid_multilab_depths \
#       --canonical-id ILIAD+50aee79f+2023-07-12-21h-13m-44s \
#       --cam2cam-json /work/nvme/bgkz/droid_annotations/cam2cam_extrinsics.json \
#       --output eef_projection.mp4


micromamba run -n gr00t python data_processing/diagnose_cross_attention.py \
    --backbone-cache-dir /work/nvme/bgkz/droid_multilab_cache_ext2 \                                                                                                  
    --depth-cache-dir /work/nvme/bgkz/droid_multilab_depth_cam2cam_ext2 \                                                                                             
    --n-batches 50 --batch-size 8 \                                      
    --checkpoints "groot_base:nvidia/GR00T-N1.6-3B" "groot_droid:nvidia/GR00T-N1.6-DROID" \                                                                           
        "my_baseline:/work/hdd/bgkz/hbhatia1/multilab_baseline_ext2/multilab-baseline-ext2" \
        "my_3d:/work/hdd/bgkz/hbhatia1/multilab_3d_cam2cam/multilab-3d-cam2cam-ext2"     