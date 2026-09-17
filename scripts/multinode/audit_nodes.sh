#!/usr/bin/env bash
# Audit every node's real GPU model against the feature label Slurm advertises.
#
# Worth doing because the labels are hand-maintained and demonstrably wrong:
# grogu-4-13 advertises ActiveFeatures=A6000 while physically holding an
# RTX 3080 Ti, and grogu-3-20 advertises rtx2080ti with Gres=gpu:8 while
# nvidia-smi reports no usable device at all. --constraint therefore cannot
# guarantee a homogeneous allocation, which is why preflight re-checks in-job.
#
# Reads /proc/driver/nvidia/gpus/*/information rather than running nvidia-smi,
# so it needs --gres=gpu:0 and one CPU. That matters: an audit that allocates a
# GPU cannot inspect a saturated node, and saturated nodes are exactly the ones
# worth checking.
#
#   ./audit_nodes.sh              # submit one tiny job per node
#   ./audit_nodes.sh --report     # print the comparison
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/logs/multinode/gpuaudit"
PARTITION="${PARTITION:-all}"
mkdir -p "$OUT"

norm() {  # crude but enough to compare a label against a marketing name
    echo "$1" | sed -e 's/NVIDIA //; s/GeForce //; s/RTX A/A/; s/Quadro //; s/ //g' \
              | tr 'A-Z' 'a-z'
}

if [[ "${1:-}" == "--report" ]]; then
    printf "%-13s %-15s %-34s %-4s %-22s %s\n" NODE SLURM_SAYS ACTUALLY_IS N HCA VERDICT
    printf '%.0s-' {1..118}; echo
    bad=0
    for f in $(ls "$OUT"/*.txt 2>/dev/null | sort); do
        node=$(basename "$f" .txt)
        real=$(grep -m1 '^MODEL ' "$f" | cut -d' ' -f2-)
        n=$(grep -m1 '^COUNT ' "$f" | cut -d' ' -f2)
        feat=$(scontrol show node "$node" 2>/dev/null | grep -oP 'ActiveFeatures=\K\S*')
        [[ -z "$real" ]] && real="(no driver info)"
        v="ok"
        if [[ "$real" == "(no driver info)" ]]; then v="*** NO GPU VISIBLE ***"
        elif [[ "$(norm "$real")" != *"$(norm "$feat")"* && "$(norm "$feat")" != *"$(norm "$real")"* ]]; then
            v="*** MISMATCH ***"
        fi
        hca=$(grep -m1 '^HCA ' "$f" | cut -d' ' -f2-)
        # NCCL_IB_HCA is one value for a whole job, so a node without mlx5_0
        # cannot join a job pinned to it -- see grogu-2-35, which has mlx4_0.
        if [[ "$hca" != *mlx5_0* ]]; then
            v="*** NO mlx5_0 ***"
        fi
        [[ "$v" != "ok" ]] && bad=$((bad+1))
        printf "%-13s %-15s %-34s %-4s %-22s %s\n" "$node" "${feat:-?}" "$real" "${n:-?}" "${hca:-?}" "$v"
    done
    echo
    echo "$bad node(s) disagree with their label — exclude them, or ask the admins to relabel."
    exit 0
fi

n=0
for node in $(scontrol show hostnames "$(sinfo -p "$PARTITION" -h -o '%N' | paste -sd,)"); do
    sbatch --parsable -J "gaudit-$node" --partition="$PARTITION" --nodelist="$node" \
        --gres=gpu:0 --cpus-per-task=1 --mem=1G --time=00:02:00 \
        --output=/dev/null --error=/dev/null \
        --wrap "for g in /proc/driver/nvidia/gpus/*/information; do \
                  grep -m1 '^Model:' \"\$g\"; done \
                | sed 's/^Model:[[:space:]]*/MODEL /' | sort -u > $OUT/$node.txt; \
                printf 'COUNT %s\n' \"\$(ls -d /proc/driver/nvidia/gpus/* 2>/dev/null | wc -l)\" \
                >> $OUT/$node.txt; \
                printf 'HCA %s\n' \"\$(ls /sys/class/infiniband 2>/dev/null | paste -sd, )\" \
                >> $OUT/$node.txt" >/dev/null 2>&1 && n=$((n+1))
done
echo "submitted $n audit jobs; when they drain: $0 --report"
