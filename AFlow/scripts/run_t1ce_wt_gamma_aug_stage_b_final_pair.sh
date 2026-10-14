#!/usr/bin/env bash
set -euo pipefail
ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="$ROOT/../sammed3d-venv/bin/python"
TRAIN="$ROOT/scripts/train_t1ce_wt_gamma_aug_stage_b_final.py"
OUT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/full972_gamma_aug_stability_20260902"
CACHE="$ROOT/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
HISTORY="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth"
EMA="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth"
MODE="${1:-formal}"
if [[ "$MODE" == "smoke" ]]; then
  for ARM in baseline_control gamma_candidate; do
    CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="/tmp/matplotlib-gamma-stageb-smoke-$ARM" "$PY" "$TRAIN" --arm "$ARM" --device cuda:0 --smoke --num-workers 0 --output-dir "$OUT/smoke_stageb_$ARM" --cache "$CACHE" --historical-teacher-checkpoint "$HISTORY" --ema-teacher-checkpoint "$EMA"
  done
  exit 0
fi
if [[ "$MODE" != "formal" ]]; then echo "usage: $0 [smoke|formal]" >&2; exit 2; fi
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-gamma-stageb-control "$PY" "$TRAIN" --arm baseline_control --device cuda:0 --output-dir "$OUT/formal_stageb_baseline_control" --cache "$CACHE" --historical-teacher-checkpoint "$HISTORY" --ema-teacher-checkpoint "$EMA" >"$OUT/stageb_baseline_control.log" 2>&1 &
PID_CONTROL=$!
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-gamma-stageb-candidate "$PY" "$TRAIN" --arm gamma_candidate --device cuda:0 --output-dir "$OUT/formal_stageb_gamma_candidate" --cache "$CACHE" --historical-teacher-checkpoint "$HISTORY" --ema-teacher-checkpoint "$EMA" >"$OUT/stageb_gamma_candidate.log" 2>&1 &
PID_CANDIDATE=$!
STATUS=0
wait "$PID_CONTROL" || STATUS=1
wait "$PID_CANDIDATE" || STATUS=1
printf '{"control_pid":%s,"candidate_pid":%s,"status":%s}\n' "$PID_CONTROL" "$PID_CANDIDATE" "$STATUS" >"$OUT/STAGE_B_PAIR_EXIT.json"
exit "$STATUS"
