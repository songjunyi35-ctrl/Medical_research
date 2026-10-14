#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="${ROOT}/../sammed3d-venv/bin/python"
WORK="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/percentile_clip_stability_20260903"
RECORD="${ROOT}/archive/research_records/2026-09-03_t1ce_percentile_clip_stability"
SPLIT="${ROOT}/data/datasets/brats2024_t1ce_percentile_clip_replications_20260903"
CONTROL_CACHE="${ROOT}/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
CANDIDATE_CACHE="/tmp/aflow_t1ce_percentile_clip_20260903/train"
CONTROL_INDEX_SHA="7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
HISTORY_TEACHER="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth"
EMA_TEACHER="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth"

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: $0 CANDIDATE_CACHE_INDEX_SHA256" >&2
  exit 2
fi
CANDIDATE_INDEX_SHA="$1"

export MPLCONFIGDIR=/tmp/aflow_mpl
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${WORK}"

for replicate in a b; do
  if [[ "${replicate}" == "a" ]]; then
    SEED=20261112
  else
    SEED=20261114
  fi
  CONTROL_OUT="${WORK}/formal_${replicate}_control"
  CANDIDATE_OUT="${WORK}/formal_${replicate}_percentile_clip"
  if [[ -e "${CONTROL_OUT}" || -e "${CANDIDATE_OUT}" ]]; then
    echo "refusing existing formal output for replicate ${replicate}" >&2
    exit 3
  fi
  "${PY}" -u scripts/train_t1ce_wt_percentile_clip_replication.py \
    --replicate "${replicate}" --arm control --cache "${CONTROL_CACHE}" \
    --expected-cache-index-sha256 "${CONTROL_INDEX_SHA}" \
    --fit-case-ids "${SPLIT}/replicate_${replicate}_fit_cases.jsonl" \
    --split-manifest "${SPLIT}/split_manifest.json" \
    --historical-teacher-checkpoint "${HISTORY_TEACHER}" \
    --ema-teacher-checkpoint "${EMA_TEACHER}" \
    --output-dir "${CONTROL_OUT}" --device cuda:0 --seed "${SEED}" \
    > "${WORK}/formal_${replicate}_control.log" 2>&1 &
  PID_CONTROL=$!
  "${PY}" -u scripts/train_t1ce_wt_percentile_clip_replication.py \
    --replicate "${replicate}" --arm percentile_clip --cache "${CANDIDATE_CACHE}" \
    --expected-cache-index-sha256 "${CANDIDATE_INDEX_SHA}" \
    --fit-case-ids "${SPLIT}/replicate_${replicate}_fit_cases.jsonl" \
    --split-manifest "${SPLIT}/split_manifest.json" \
    --historical-teacher-checkpoint "${HISTORY_TEACHER}" \
    --ema-teacher-checkpoint "${EMA_TEACHER}" \
    --output-dir "${CANDIDATE_OUT}" --device cuda:1 --seed "${SEED}" \
    > "${WORK}/formal_${replicate}_percentile_clip.log" 2>&1 &
  PID_CANDIDATE=$!
  set +e
  wait "${PID_CONTROL}"
  RC_CONTROL=$?
  wait "${PID_CANDIDATE}"
  RC_CANDIDATE=$?
  set -e
  if [[ "${RC_CONTROL}" -ne 0 || "${RC_CANDIDATE}" -ne 0 ]]; then
    echo "replicate ${replicate} exits: control=${RC_CONTROL} candidate=${RC_CANDIDATE}" >&2
    exit 4
  fi
done

"${PY}" scripts/freeze_t1ce_wt_percentile_clip_inner_preaccess.py \
  --work-dir "${WORK}" --split-manifest "${SPLIT}/split_manifest.json" \
  --control-cache "${CONTROL_CACHE}" --candidate-cache "${CANDIDATE_CACHE}" \
  --output "${RECORD}/INNER_PREACCESS.json"
PREACCESS_SHA=$(sha256sum "${RECORD}/INNER_PREACCESS.json" | cut -d' ' -f1)

"${PY}" scripts/evaluate_t1ce_wt_percentile_clip_replication_inner.py \
  --work-dir "${WORK}" --split-manifest "${SPLIT}/split_manifest.json" \
  --control-cache "${CONTROL_CACHE}" --candidate-cache "${CANDIDATE_CACHE}" \
  --preaccess "${RECORD}/INNER_PREACCESS.json" \
  --expected-preaccess-sha256 "${PREACCESS_SHA}" \
  --output "${WORK}/STAGE_A_INNER.json" --device cuda:0 --num-workers 4

echo "percentile_clip_stage_a_complete"
