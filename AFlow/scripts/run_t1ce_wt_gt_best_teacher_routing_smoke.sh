#!/usr/bin/env bash
set -uo pipefail

ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="$ROOT/../sammed3d-venv/bin/python"
OUT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/gt_best_teacher_routing_20260901"
TRAIN="$ROOT/scripts/train_t1ce_wt_gt_best_teacher_routing.py"
CACHE="$ROOT/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
MANIFEST="$ROOT/data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
T32="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth"
T40="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/full972_multiseed_20260825/formal_seed20260940/plain_full972_seed20260940_epoch168.pth"
T41="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/full972_multiseed_20260825/formal_seed20260941/plain_full972_seed20260941_epoch168.pth"

mkdir -p "$OUT"
cd "$ROOT"
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-routing-smoke-control "$PY" "$TRAIN" \
  --arm anchor_repeat3_control --cache "$CACHE" --train-manifest "$MANIFEST" \
  --teacher-seed32 "$T32" --teacher-seed40 "$T40" --teacher-seed41 "$T41" \
  --output-dir "$OUT/smoke_anchor_repeat3_control" --device cuda:0 \
  --smoke --smoke-steps 2 --limit-train-cases 8 \
  > "$OUT/smoke_anchor_repeat3_control.log" 2>&1 &
PID0=$!
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-routing-smoke-candidate "$PY" "$TRAIN" \
  --arm gt_best_of3_candidate --cache "$CACHE" --train-manifest "$MANIFEST" \
  --teacher-seed32 "$T32" --teacher-seed40 "$T40" --teacher-seed41 "$T41" \
  --output-dir "$OUT/smoke_gt_best_of3_candidate" --device cuda:1 \
  --smoke --smoke-steps 2 --limit-train-cases 8 \
  > "$OUT/smoke_gt_best_of3_candidate.log" 2>&1 &
PID1=$!
wait "$PID0"; STATUS0=$?
wait "$PID1"; STATUS1=$?
printf '{"control_exit":%d,"candidate_exit":%d}\n' "$STATUS0" "$STATUS1" > "$OUT/SMOKE_EXIT.json"
test "$STATUS0" -eq 0
test "$STATUS1" -eq 0
