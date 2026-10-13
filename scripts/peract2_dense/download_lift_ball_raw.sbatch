#!/usr/bin/env bash
# Downloads the original PerAct2 raw bimanual_lift_ball train/val (all frames), the demos behind Peract2_zarr.
#SBATCH --job-name=p2dense_download
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-3-5,grogu-4-18,grogu-1-20,grogu-4-23
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=23:59:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/data/p2dense_download-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/data/p2dense_download-%j.err

set -euo pipefail
TASK="${TASK:-bimanual_lift_ball}"
RAW_ROOT="${RAW_ROOT:-/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/raw}"
LINK="https://dataset.cs.washington.edu/fox/bimanual/image_size_256"
UNSQUASHFS="${UNSQUASHFS:-/home/harshilb/miniconda3/bin/unsquashfs}"
mkdir -p "$RAW_ROOT/_squashfs"
cd "$RAW_ROOT/_squashfs"
wget -q -O SHA256SUM "$LINK/SHA256SUM"
for SPLIT in train val; do
    DST="$RAW_ROOT/$SPLIT/$TASK"
    if [ -d "$DST" ]; then echo "[SKIP] $DST exists"; continue; fi
    F="$TASK.$SPLIT.squashfs"
    echo "[download] $F host=$(hostname)"
    wget -q -c --no-check-certificate -O "$F" "$LINK/$F"
    grep " $F\$" SHA256SUM | sha256sum -c -
    mkdir -p "$RAW_ROOT/$SPLIT"
    "$UNSQUASHFS" -q -d "$DST.tmp" "$F"
    mv "$DST.tmp" "$DST"
    rm "$F"
    echo "[done] $DST episodes=$(ls "$DST/all_variations/episodes" | wc -l)"
done
