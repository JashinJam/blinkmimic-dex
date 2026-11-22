#!/bin/bash
# Train the causal Sharpa path with the released algorithm recipe and eager collection.
# The object reference entity writes a mocap pose every step, so CUDA collection/reset graphs
# are intentionally omitted. All algorithmic flags from PRESET[main] remain explicit here.
# Usage: bash scripts/train_sharpa.sh <log_dir> [extra train.py flags]
set -eu
cd "$(dirname "$0")/.."
LOG_DIR=${1:?usage: bash scripts/train_sharpa.sh <log_dir> [extra train.py flags]}; shift
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

PYTHONPATH=src ${PYTHON:-uv run python} -u -m blinkmimic.train --log_dir "$LOG_DIR" \
  --robot sharpa \
  --causal_goal \
  --pi_ema 0.995 \
  --no_sync \
  --fused_adam \
  --fused_rot \
  --compile_more \
  --lr 4e-4 \
  --wm_batches 5 \
  --wm_hidden 1024,1024,1024 \
  --fail_reset 0.3 \
  --contact_reward 0.25 \
  --wm_structured \
  --wm_mirror 0.5 \
  --pi_opt muon \
  --wm_opt muon \
  --pi_ref_skip \
  --wm_only_iters 0 \
  --bin_fix \
  --ref_contact_fix \
  --pi_batch 2048 \
  --pi_lr_mult 4 \
  --lr_decay_ret 0.75 \
  --lr_decay_len 300 \
  --iterations 3500 \
  --ckpt_every 100 \
  --ckpt_light \
  --eprobe_share 0.15 \
  --eprobe_start 1000 \
  --eprobe_every 125 \
  --eprobe_decay 0.7 \
  --lr_decay_by 1400 \
  "$@" 2>&1 | tee "$LOG_DIR/stdout.log"
