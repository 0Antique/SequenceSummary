#!/bin/bash
set -euo pipefail

# Batch-generate JSON outputs for all CSVs under ./events2
# and write them into: ./visualization/app/public/assets

python RunAll2.py \
  --input_dir "./events2" \
  --pattern "*.csv" \
  --attr "event" \
  --grpattr "traj_id" \
  --startidx 2 \
  "$@"
