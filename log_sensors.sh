#!/bin/bash

LOG_FILE="${1:-sensors_$(date +%Y%m%d_%H%M%S).log}"

echo "Logging sensors to: $LOG_FILE"
echo "Press Ctrl+C to stop."

while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"
    sensors >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    sleep 1
done
