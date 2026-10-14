#!/usr/bin/env bash
set -uo pipefail
ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="$ROOT/../sammed3d-venv/bin/python"
OUT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824"
TRAIN="$ROOT/scripts/train_t1ce_wt_voxel_kd_fulltrajectory_final.py"
CACHE="$ROOT/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
HISTORY="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth"
EMA="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth"
cd "$ROOT"
PYTHONDONTWRITEBYTECODE=1 "$PY" "$TRAIN" --arm full756_control --cache "$CACHE" \
  --train-case-ids data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/train_cases.jsonl \
  --historical-teacher-checkpoint "$HISTORY" --ema-teacher-checkpoint "$EMA" \
  --output-dir "$OUT/final_formal_full756_control" --device cuda:0 > "$OUT/final756_control.log" 2>&1 &
P0=$!
PYTHONDONTWRITEBYTECODE=1 "$PY" "$TRAIN" --arm full972_candidate --cache "$CACHE" \
  --train-case-ids data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl \
  --historical-teacher-checkpoint "$HISTORY" --ema-teacher-checkpoint "$EMA" \
  --output-dir "$OUT/final_formal_full972_candidate" --device cuda:1 > "$OUT/final972_candidate.log" 2>&1 &
P1=$!
wait "$P0"; S0=$?
wait "$P1"; S1=$?
printf '{"control_exit":%d,"candidate_exit":%d}\n' "$S0" "$S1" > "$OUT/FINAL_PAIR_EXIT.json"
test "$S0" -eq 0
test "$S1" -eq 0
