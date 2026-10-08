#!/bin/bash
cd /home/harshilb/3dfa_unified
for i in $(seq 1 600); do
  PYTHONPATH=$PWD /home/harshilb/miniconda3/envs/3dfa/bin/python scripts/analysis/watch_extractor_alive.py peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch peract2_orbital_vid_pooled_randhist_deltam_inputskip_scratch >> logs/analysis/extractor_alive.stderr 2>&1
  n=$(grep -cE " (200)k " logs/analysis/extractor_alive.log 2>/dev/null); [[ "$n" -ge 2 ]] && break
  sleep 600
done
