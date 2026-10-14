#!/usr/bin/env bash
set -uo pipefail

ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="$ROOT/../sammed3d-venv/bin/python"
OUT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831"
TRAIN="$ROOT/scripts/train_t1ce_wt_gcsam_inner.py"
CACHE="$ROOT/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
HISTORY="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth"
EMA="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth"

mkdir -p "$OUT"
cd "$ROOT"
PYTHONDONTWRITEBYTECODE=1 "$PY" "$TRAIN" --arm adamw_control --cache "$CACHE" \
  --historical-teacher-checkpoint "$HISTORY" --ema-teacher-checkpoint "$EMA" \
  --output-dir "$OUT/inner_formal_adamw_control" --device cuda:0 \
  > "$OUT/inner_adamw_control.log" 2>&1 &
PID0=$!
PYTHONDONTWRITEBYTECODE=1 "$PY" "$TRAIN" --arm gcsam_candidate --cache "$CACHE" \
  --historical-teacher-checkpoint "$HISTORY" --ema-teacher-checkpoint "$EMA" \
  --output-dir "$OUT/inner_formal_gcsam_candidate" --device cuda:1 \
  > "$OUT/inner_gcsam_candidate.log" 2>&1 &
PID1=$!
wait "$PID0"; STATUS0=$?
wait "$PID1"; STATUS1=$?
printf '{"adamw_control_exit":%d,"gcsam_candidate_exit":%d}\n' "$STATUS0" "$STATUS1" > "$OUT/INNER_PAIR_EXIT.json"
test "$STATUS0" -eq 0
test "$STATUS1" -eq 0
