#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="${ROOT}/../sammed3d-venv/bin/python"
WORK="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/gcsam_fixed184_replication_20260903"
RECORD="${ROOT}/archive/research_records/2026-09-03_gcsam_fixed184_replication"
SPLIT="${ROOT}/data/datasets/brats2024_t1ce_gcsam_fixed184_replications_20260903"
CACHE="${ROOT}/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
HISTORY_TEACHER="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth"
EMA_TEACHER="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth"

export MPLCONFIGDIR=/tmp/aflow_mpl
export PYTHONDONTWRITEBYTECODE=1

while [[ ! -f "${WORK}/formal_a_adamw_control/TRAIN_DONE.json" || ! -f "${WORK}/formal_b_adamw_control/TRAIN_DONE.json" ]]; do
  sleep 55
done

cd "${ROOT}"
"${PY}" -c 'from pathlib import Path; import scripts.evaluate_t1ce_wt_gcsam_fixed184_replication_inner as e; w=Path("workspace/BraTS/sammed3d_t1ce_wt/gcsam_fixed184_replication_20260903"); e.load_training_report(w/"formal_a_adamw_control/TRAIN_DONE.json","a","adamw_control"); e.load_training_report(w/"formal_b_adamw_control/TRAIN_DONE.json","b","adamw_control"); print("controls_preflight_passed")'

if [[ -e "${WORK}/formal_a_gcsam_candidate" || -e "${WORK}/formal_b_gcsam_candidate" ]]; then
  echo "refusing existing formal candidate directory" >&2
  exit 2
fi

"${PY}" -u scripts/train_t1ce_wt_gcsam_fixed184_replication.py \
  --replicate a --arm gcsam_candidate --cache "${CACHE}" \
  --fit-case-ids "${SPLIT}/replicate_a_fit_cases.jsonl" \
  --historical-teacher-checkpoint "${HISTORY_TEACHER}" \
  --ema-teacher-checkpoint "${EMA_TEACHER}" \
  --output-dir "${WORK}/formal_a_gcsam_candidate" \
  --device cuda:0 --seed 20261102 \
  > "${WORK}/formal_a_gcsam_candidate.log" 2>&1 &
PID_A=$!

"${PY}" -u scripts/train_t1ce_wt_gcsam_fixed184_replication.py \
  --replicate b --arm gcsam_candidate --cache "${CACHE}" \
  --fit-case-ids "${SPLIT}/replicate_b_fit_cases.jsonl" \
  --historical-teacher-checkpoint "${HISTORY_TEACHER}" \
  --ema-teacher-checkpoint "${EMA_TEACHER}" \
  --output-dir "${WORK}/formal_b_gcsam_candidate" \
  --device cuda:1 --seed 20261104 \
  > "${WORK}/formal_b_gcsam_candidate.log" 2>&1 &
PID_B=$!

set +e
wait "${PID_A}"
RC_A=$?
wait "${PID_B}"
RC_B=$?
set -e
if [[ "${RC_A}" -ne 0 || "${RC_B}" -ne 0 ]]; then
  echo "candidate exits: ${RC_A} ${RC_B}" >&2
  exit 3
fi

"${PY}" scripts/freeze_t1ce_wt_gcsam_fixed184_inner_preaccess.py \
  --work-dir "${WORK}" --output "${RECORD}/INNER_PREACCESS.json"
PREACCESS_SHA=$(sha256sum "${RECORD}/INNER_PREACCESS.json" | cut -d' ' -f1)

"${PY}" scripts/evaluate_t1ce_wt_gcsam_fixed184_replication_inner.py \
  --work-dir "${WORK}" --cache "${CACHE}" \
  --preaccess "${RECORD}/INNER_PREACCESS.json" \
  --expected-preaccess-sha256 "${PREACCESS_SHA}" \
  --output "${WORK}/FIXED184_REPLICATION_INNER.json" \
  --device cuda:0 --num-workers 4

echo "stage_a_evaluation_complete"
