#!/usr/bin/env bash
set -uo pipefail

ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="$ROOT/../sammed3d-venv/bin/python"
OUT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/fresh_multiseed_consensus_20260901"
TRAIN="$ROOT/scripts/train_t1ce_wt_consensus_seed_teacher.py"
CACHE="$ROOT/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
HISTORY="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth"
EMA="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth"

mkdir -p "$OUT"
cd "$ROOT"
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-consensus-smoke31 "$PY" "$TRAIN" \
  --arm teacher_seed20261031 --cache "$CACHE" --historical-teacher-checkpoint "$HISTORY" \
  --ema-teacher-checkpoint "$EMA" --output-dir "$OUT/smoke_teacher_seed20261031" \
  --device cuda:0 --smoke --smoke-steps 2 --limit-fit-cases 8 \
  > "$OUT/smoke_teacher_seed20261031.log" 2>&1 &
PID0=$!
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-consensus-smoke32 "$PY" "$TRAIN" \
  --arm teacher_seed20261032 --cache "$CACHE" --historical-teacher-checkpoint "$HISTORY" \
  --ema-teacher-checkpoint "$EMA" --output-dir "$OUT/smoke_teacher_seed20261032" \
  --device cuda:1 --smoke --smoke-steps 2 --limit-fit-cases 8 \
  > "$OUT/smoke_teacher_seed20261032.log" 2>&1 &
PID1=$!
wait "$PID0"; STATUS0=$?
wait "$PID1"; STATUS1=$?

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-consensus-smoke33 "$PY" "$TRAIN" \
  --arm teacher_seed20261033 --cache "$CACHE" --historical-teacher-checkpoint "$HISTORY" \
  --ema-teacher-checkpoint "$EMA" --output-dir "$OUT/smoke_teacher_seed20261033" \
  --device cuda:0 --smoke --smoke-steps 2 --limit-fit-cases 8 \
  > "$OUT/smoke_teacher_seed20261033.log" 2>&1
STATUS2=$?
printf '{"seed31_exit":%d,"seed32_exit":%d,"seed33_exit":%d}\n' "$STATUS0" "$STATUS1" "$STATUS2" > "$OUT/TEACHER_SMOKE_EXIT.json"
test "$STATUS0" -eq 0
test "$STATUS1" -eq 0
test "$STATUS2" -eq 0
