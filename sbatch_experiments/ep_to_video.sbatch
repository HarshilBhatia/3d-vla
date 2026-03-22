#!/bin/bash
#SBATCH --job-name=ep_to_video
#SBATCH --partition=shubhamlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j-ep_to_video.out
#SBATCH --error=logs/%j-ep_to_video.err

# Convert a RLBench episode directory to a tiled MP4 video.
#
# Usage examples:
#   # Single episode, default output (episode.mp4 inside episode dir):
#   sbatch sbatch_experiments/ep_to_video.sbatch \
#       peract_raw_rot10/place_cups/all_variations/episodes/episode0
#
#   # Specify output path:
#   OUTFILE=/tmp/place_cups_ep0.mp4 sbatch sbatch_experiments/ep_to_video.sbatch \
#       peract_raw_rot10/place_cups/all_variations/episodes/episode0
#
#   # All episodes of a task (creates episode0.mp4, episode1.mp4, ...):
#   EPISODE_GLOB="peract_raw_rot10/close_jar/all_variations/episodes/episode*" \
#       sbatch sbatch_experiments/ep_to_video.sbatch
#
#   # Only certain cameras as a horizontal strip:
#   CAMERAS="front_rgb left_shoulder_rgb" sbatch sbatch_experiments/ep_to_video.sbatch \
#       peract_raw_rot10/place_cups/all_variations/episodes/episode0

set -eo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR

REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
cd "$REPO_DIR"
mkdir -p logs

CONDA_BASE="${CONDA_BASE:-/home/lzaceria/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate 3dfa

FPS="${FPS:-10}"
CAMERAS_FLAG=""
if [[ -n "${CAMERAS:-}" ]]; then
    CAMERAS_FLAG="--cameras $CAMERAS"
fi

# ---- Mode 1: single episode dir passed as $1 or $EPISODE_DIR ----
if [[ -n "${1:-}" ]] || [[ -n "${EPISODE_DIR:-}" ]]; then
    EP_DIR="${1:-$EPISODE_DIR}"
    [[ "${EP_DIR}" != /* ]] && EP_DIR="$REPO_DIR/$EP_DIR"
    OUTFILE="${OUTFILE:-${EP_DIR}/episode.mp4}"
    echo "Episode : $EP_DIR"
    echo "Output  : $OUTFILE"
    python scripts/rlbench/episode_to_video.py \
        --episode_dir "$EP_DIR" \
        --output "$OUTFILE" \
        --fps "$FPS" \
        $CAMERAS_FLAG

# ---- Mode 2: glob over multiple episodes ----
elif [[ -n "${EPISODE_GLOB:-}" ]]; then
    for ep in $EPISODE_GLOB; do
        [[ -d "$ep" ]] || continue
        [[ "${ep}" != /* ]] && ep="$REPO_DIR/$ep"
        ep_name="$(basename $ep)"
        out_dir="$(dirname $ep)"
        OUTFILE="$out_dir/${ep_name}.mp4"
        echo "Episode : $ep  ->  $OUTFILE"
        python scripts/rlbench/episode_to_video.py \
            --episode_dir "$ep" \
            --output "$OUTFILE" \
            --fps "$FPS" \
            $CAMERAS_FLAG
    done

else
    echo "[ERROR] Provide an episode dir as \$1, EPISODE_DIR=..., or EPISODE_GLOB=..."
    exit 1
fi

echo "Done at $(date)"
