#!/usr/bin/env bash
set -uo pipefail

ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="$ROOT/../sammed3d-venv/bin/python"
OUT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824"
TRAIN="$ROOT/scripts/train_t1ce_wt_voxel_kd_fulltrajectory_inner.py"
CACHE="$ROOT/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
SPLIT="$ROOT/data/datasets/brats2024_t1ce_fulltrajectory_common_inner_seed20260930_v1"
HISTORY="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth"
EMA="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth"

cd "$ROOT"
PYTHONDONTWRITEBYTECODE=1 "$PY" "$TRAIN" \
  --arm inner648_control \
  --cache "$CACHE" \
  --fit-case-ids "$SPLIT/control_fit_cases.jsonl" \
  --historical-teacher-checkpoint "$HISTORY" \
  --ema-teacher-checkpoint "$EMA" \
  --output-dir "$OUT/inner_formal_inner648_control" \
  --device cuda:0 > "$OUT/inner648_control.log" 2>&1 &
PID0=$!

PYTHONDONTWRITEBYTECODE=1 "$PY" "$TRAIN" \
  --arm inner864_candidate \
  --cache "$CACHE" \
  --fit-case-ids "$SPLIT/candidate_fit_cases.jsonl" \
  --historical-teacher-checkpoint "$HISTORY" \
  --ema-teacher-checkpoint "$EMA" \
  --output-dir "$OUT/inner_formal_inner864_candidate" \
  --device cuda:1 > "$OUT/inner864_candidate.log" 2>&1 &
PID1=$!

wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
printf '{"control_exit":%d,"candidate_exit":%d}\n' "$STATUS0" "$STATUS1" > "$OUT/INNER_PAIR_EXIT.json"
test "$STATUS0" -eq 0
test "$STATUS1" -eq 0
