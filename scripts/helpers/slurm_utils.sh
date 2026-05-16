#!/bin/bash
# Shared helpers for slurm training scripts.

# Copy a zarr (or any directory) from grogu to /scratch if the destination doesn't exist.
# Usage: rsync_if_missing <src> <dst>
rsync_if_missing() {
    local src=$1 dst=$2
    if [ ! -d "$dst" ]; then
        echo "Copying $(basename "$src") → $dst"
        mkdir -p "$(dirname "$dst")"
        rsync -a --info=progress2 "$src" "$(dirname "$dst")/"
    else
        echo "Scratch copy exists, skipping: $dst"
    fi
}

# Remap a /grogu/user/harshilb path to /scratch/harshilb and rsync if missing.
# Sets the variable named by $1 to the scratch path.
# Usage: to_scratch train_data_dir   (modifies the variable in the caller's scope)
to_scratch() {
    local varname=$1
    local src=${!varname}
    local dst=${src/\/grogu\/user\/harshilb/\/scratch\/harshilb}
    rsync_if_missing "$src" "$dst"
    printf -v "$varname" '%s' "$dst"
}
