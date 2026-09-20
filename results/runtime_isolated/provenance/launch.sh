#!/usr/bin/env bash
set -uo pipefail
cd /workspace/residential-experiment || exit 1
log_dir=/workspace/setup-logs/runtime-isolated
mkdir -p "$log_dir"
exec > "$log_dir/run.log" 2>&1
date -u
export PYTHONUNBUFFERED=1
timeout --signal=TERM --kill-after=60s 6h \
  /opt/residential-venv/bin/python -m fl_pipeline.run_experiment \
  --feature-set baseline_16 --model-size small \
  --device cpu --num-threads 1 \
  --rounds 20 --local-epochs 5 --batch-size 256 \
  --split stratified --results-dir runs/runtime_isolated \
  --seeds 42 \
  --configs baseline_fl fs_mild fs_aggressive dp_sgd feddpa
status=$?
date -u
printf '%s\n' "$status" > "$log_dir/run.exit"
exit "$status"
