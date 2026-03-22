#!/bin/bash


for i in 51 102 202 303 506; do
    micromamba run -n gr00t python data_processing/droid_visualiser.py \
        --dataset-dir /work/nvme/bgkz/droid_raw_large_superset \
        --raw-dir /work/nvme/bgkz/droid_multilab_raw \
        --depth-dir /work/nvme/bgkz/droid_multilab_depths \
        --episode-idx "$i" \
        --max-timesteps "${MAX_TIMESTEPS:-30}" \
        --grpc-port "${GRPC_PORT:-9876}" \
        --web-port "${WEB_PORT:-9090}"
done