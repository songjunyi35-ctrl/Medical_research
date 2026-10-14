#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="$ROOT/../sammed3d-venv/bin/python"
TRAIN="$ROOT/scripts/train_t1ce_wt_lawa6_stage_a.py"
OUT="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/full972_lawa6_stability_20260902"
CACHE="$ROOT/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
HISTORY="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth"
EMA="$ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth"
MODE="${1:-formal}"

if [[ "$MODE" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-lawa6-smoke61 "$PY" "$TRAIN" \
    --arm stagea_seed20261061 --device cuda:0 --smoke --num-workers 0 \
    --output-dir "$OUT/smoke_stagea_seed20261061" --cache "$CACHE" \
    --historical-teacher-checkpoint "$HISTORY" --ema-teacher-checkpoint "$EMA"
  exit 0
fi
if [[ "$MODE" != "formal" ]]; then
  echo "usage: $0 [smoke|formal]" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-lawa6-formal61 "$PY" "$TRAIN" \
  --arm stagea_seed20261061 --device cuda:0 --output-dir "$OUT/formal_stagea_seed20261061" \
  --cache "$CACHE" --historical-teacher-checkpoint "$HISTORY" --ema-teacher-checkpoint "$EMA" \
  >"$OUT/stagea_seed20261061.log" 2>&1 &
PID61=$!
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib-lawa6-formal62 "$PY" "$TRAIN" \
  --arm stagea_seed20261062 --device cuda:0 --output-dir "$OUT/formal_stagea_seed20261062" \
  --cache "$CACHE" --historical-teacher-checkpoint "$HISTORY" --ema-teacher-checkpoint "$EMA" \
  >"$OUT/stagea_seed20261062.log" 2>&1 &
PID62=$!

STATUS=0
wait "$PID61" || STATUS=1
wait "$PID62" || STATUS=1
printf '{"seed61_pid":%s,"seed62_pid":%s,"status":%s}\n' "$PID61" "$PID62" "$STATUS" >"$OUT/STAGE_A_PAIR_EXIT.json"
exit "$STATUS"
