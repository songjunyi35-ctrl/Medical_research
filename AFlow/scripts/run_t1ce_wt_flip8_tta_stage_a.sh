#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="${ROOT}/../sammed3d-venv/bin/python"
WORK="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/flip8_tta_stability_20260904"
RECORD="${ROOT}/archive/research_records/2026-09-04_direct_locator_flip8_tta_stability"
SPLIT="${ROOT}/data/datasets/brats2024_t1ce_flip8_tta_replications_20260904"
CACHE="${ROOT}/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
CACHE_INDEX_SHA="7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
HISTORY_TEACHER="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth"
EMA_TEACHER="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth"

export MPLCONFIGDIR=/tmp/aflow_mpl
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${WORK}"

if [[ -e "${WORK}/formal_a_base" || -e "${WORK}/formal_b_base" ]]; then
  echo "refusing existing formal output" >&2
  exit 2
fi

"${PY}" -u scripts/train_t1ce_wt_flip8_tta_replication.py \
  --replicate a --arm base --cache "${CACHE}" \
  --expected-cache-index-sha256 "${CACHE_INDEX_SHA}" \
  --fit-case-ids "${SPLIT}/replicate_a_fit_cases.jsonl" \
  --split-manifest "${SPLIT}/split_manifest.json" \
  --historical-teacher-checkpoint "${HISTORY_TEACHER}" \
  --ema-teacher-checkpoint "${EMA_TEACHER}" \
  --output-dir "${WORK}/formal_a_base" --device cuda:0 --seed 20261202 \
  > "${WORK}/formal_a_base.log" 2>&1 &
PID_A=$!

"${PY}" -u scripts/train_t1ce_wt_flip8_tta_replication.py \
  --replicate b --arm base --cache "${CACHE}" \
  --expected-cache-index-sha256 "${CACHE_INDEX_SHA}" \
  --fit-case-ids "${SPLIT}/replicate_b_fit_cases.jsonl" \
  --split-manifest "${SPLIT}/split_manifest.json" \
  --historical-teacher-checkpoint "${HISTORY_TEACHER}" \
  --ema-teacher-checkpoint "${EMA_TEACHER}" \
  --output-dir "${WORK}/formal_b_base" --device cuda:1 --seed 20261204 \
  > "${WORK}/formal_b_base.log" 2>&1 &
PID_B=$!

set +e
wait "${PID_A}"
RC_A=$?
wait "${PID_B}"
RC_B=$?
set -e
if [[ "${RC_A}" -ne 0 || "${RC_B}" -ne 0 ]]; then
  echo "training exits: a=${RC_A} b=${RC_B}" >&2
  exit 3
fi

"${PY}" scripts/freeze_t1ce_wt_flip8_tta_inner_preaccess.py \
  --work-dir "${WORK}" --split-manifest "${SPLIT}/split_manifest.json" \
  --cache "${CACHE}" --output "${RECORD}/INNER_PREACCESS.json"
PREACCESS_SHA=$(sha256sum "${RECORD}/INNER_PREACCESS.json" | cut -d' ' -f1)

"${PY}" scripts/evaluate_t1ce_wt_flip8_tta_replication_inner.py \
  --work-dir "${WORK}" --split-manifest "${SPLIT}/split_manifest.json" \
  --cache "${CACHE}" --preaccess "${RECORD}/INNER_PREACCESS.json" \
  --expected-preaccess-sha256 "${PREACCESS_SHA}" \
  --output "${WORK}/STAGE_A_INNER.json" --device cuda:0 --num-workers 4

echo "flip8_tta_stage_a_complete"
