#!/bin/bash

# ─────────────────────────────────────────
#  GPU Monitor Script - NVIDIA
#  Logs: temp, fan speed, GPU/mem usage,
#        power draw, and full nvidia-smi
# ─────────────────────────────────────────

LOG_FILE="gpu_monitor_$(date +%Y%m%d_%H%M%S).log"
INTERVAL=5  # seconds between each reading

echo "========================================" | tee -a "$LOG_FILE"
echo " NVIDIA GPU Monitor Started"             | tee -a "$LOG_FILE"
echo " Log file : $LOG_FILE"                   | tee -a "$LOG_FILE"
echo " Interval : every ${INTERVAL}s"          | tee -a "$LOG_FILE"
echo " Press Ctrl+C to stop"                   | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # ── Quick one-line stats ──────────────────────────────────────────
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null)
    FAN=$(nvidia-smi --query-gpu=fan.speed \
        --format=csv,noheader,nounits 2>/dev/null)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null)
    MEM_UTIL=$(nvidia-smi --query-gpu=utilization.memory \
        --format=csv,noheader,nounits 2>/dev/null)
    MEM_USED=$(nvidia-smi --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null)
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total \
        --format=csv,noheader,nounits 2>/dev/null)
    POWER=$(nvidia-smi --query-gpu=power.draw \
        --format=csv,noheader,nounits 2>/dev/null)

    echo "──────────────────────────────────────────" | tee -a "$LOG_FILE"
    echo "[$TIMESTAMP]"                               | tee -a "$LOG_FILE"
    echo "  🌡  Temperature : ${TEMP} °C"             | tee -a "$LOG_FILE"
    echo "  🌀  Fan Speed   : ${FAN} %"               | tee -a "$LOG_FILE"
    echo "  ⚡  GPU Usage   : ${GPU_UTIL} %"          | tee -a "$LOG_FILE"
    echo "  💾  Mem Usage   : ${MEM_USED} / ${MEM_TOTAL} MiB (${MEM_UTIL} %)" | tee -a "$LOG_FILE"
    echo "  🔌  Power Draw  : ${POWER} W"             | tee -a "$LOG_FILE"

    # ── Full nvidia-smi snapshot (every 60s) ─────────────────────────
    if (( $(date +%s) % 60 < INTERVAL )); then
        echo "" | tee -a "$LOG_FILE"
        echo "  [Full nvidia-smi snapshot]" | tee -a "$LOG_FILE"
        nvidia-smi 2>/dev/null | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"


    sleep "$INTERVAL"
done