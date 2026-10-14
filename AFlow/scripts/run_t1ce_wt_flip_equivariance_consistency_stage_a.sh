#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/songjunyi.666/playground/AFlow"
PY="${ROOT}/../sammed3d-venv/bin/python"
WORK="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/flip_equivariance_consistency_20260904"
RECORD="${ROOT}/archive/research_records/2026-09-04_flip_equivariance_consistency_stability"
SPLIT="${ROOT}/data/datasets/brats2024_t1ce_flip_equivariance_consistency_20260904"
CACHE="${ROOT}/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
CACHE_INDEX_SHA="7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
HISTORY_TEACHER="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth"
EMA_TEACHER="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth"
TRAIN="${ROOT}/scripts/train_t1ce_wt_flip_equivariance_consistency.py"

export MPLCONFIGDIR=/tmp/aflow_mpl
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${WORK}"

for replicate in c d; do
  for arm in control_w0 candidate_w01; do
    if [[ -e "${WORK}/formal_${replicate}_${arm}" ]]; then
      echo "refusing existing formal output: ${replicate}/${arm}" >&2
      exit 2
    fi
  done
done
if [[ -e "${RECORD}/INNER_PREACCESS.json" || -e "${WORK}/STAGE_A_INNER.json" || -e "${WORK}/STAGE_A_INNER.json.started.json" ]]; then
  echo "refusing existing inner access artifact" >&2
  exit 2
fi

run_pair() {
  local replicate="$1"
  local seed="$2"

  "${PY}" -u "${TRAIN}" \
    --replicate "${replicate}" --arm control_w0 --cache "${CACHE}" \
    --expected-cache-index-sha256 "${CACHE_INDEX_SHA}" \
    --fit-case-ids "${SPLIT}/replicate_${replicate}_fit_cases.jsonl" \
    --split-manifest "${SPLIT}/split_manifest.json" \
    --historical-teacher-checkpoint "${HISTORY_TEACHER}" \
    --ema-teacher-checkpoint "${EMA_TEACHER}" \
    --output-dir "${WORK}/formal_${replicate}_control_w0" \
    --device cuda:0 --seed "${seed}" \
    > "${WORK}/formal_${replicate}_control_w0.log" 2>&1 &
  local pid_control=$!

  "${PY}" -u "${TRAIN}" \
    --replicate "${replicate}" --arm candidate_w01 --cache "${CACHE}" \
    --expected-cache-index-sha256 "${CACHE_INDEX_SHA}" \
    --fit-case-ids "${SPLIT}/replicate_${replicate}_fit_cases.jsonl" \
    --split-manifest "${SPLIT}/split_manifest.json" \
    --historical-teacher-checkpoint "${HISTORY_TEACHER}" \
    --ema-teacher-checkpoint "${EMA_TEACHER}" \
    --output-dir "${WORK}/formal_${replicate}_candidate_w01" \
    --device cuda:1 --seed "${seed}" \
    > "${WORK}/formal_${replicate}_candidate_w01.log" 2>&1 &
  local pid_candidate=$!

  set +e
  wait "${pid_control}"
  local rc_control=$?
  wait "${pid_candidate}"
  local rc_candidate=$?
  set -e
  if [[ "${rc_control}" -ne 0 || "${rc_candidate}" -ne 0 ]]; then
    echo "pair ${replicate} exits: control=${rc_control} candidate=${rc_candidate}" >&2
    exit 3
  fi
}

run_pair c 20261212
run_pair d 20261214

"${PY}" "${ROOT}/scripts/freeze_t1ce_wt_flip_equivariance_consistency_inner_preaccess.py" \
  --work-dir "${WORK}" --split-manifest "${SPLIT}/split_manifest.json" \
  --cache "${CACHE}" --output "${RECORD}/INNER_PREACCESS.json"
PREACCESS_SHA=$(sha256sum "${RECORD}/INNER_PREACCESS.json" | cut -d' ' -f1)

"${PY}" "${ROOT}/scripts/evaluate_t1ce_wt_flip_equivariance_consistency_inner.py" \
  --work-dir "${WORK}" --split-manifest "${SPLIT}/split_manifest.json" \
  --cache "${CACHE}" --preaccess "${RECORD}/INNER_PREACCESS.json" \
  --expected-preaccess-sha256 "${PREACCESS_SHA}" \
  --output "${WORK}/STAGE_A_INNER.json" --device cuda:0 --num-workers 4

echo "flip_equivariance_consistency_stage_a_complete"
