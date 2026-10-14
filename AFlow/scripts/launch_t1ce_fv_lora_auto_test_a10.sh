#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
TRAIN_REL="workspace/BraTS/sammed3d_t1ce_wt/fv_lora_r4_aug_train972_val108_e50_s20260725"
BASE="${ROOT}/workspace/BraTS/sammed3d_t1ce_wt"
WATCH_LOG="${BASE}/fv_lora_r4_aug_auto_test_20260723.watch.log"
WATCH_PID="${BASE}/fv_lora_r4_aug_auto_test_20260723.watch.pid"
STATE="${BASE}/AUTO_TEST_WATCHER_FV_LORA_AUG_20260723.json"
FREEZE="${BASE}/FROZEN_FV_LORA_AUG_TEST_CONFIGURATION_20260723.json"
REPORT="${BASE}/FROZEN_fv_lora_aug_test270_cumulative3_s20260723.json"
FINAL="${BASE}/FINAL_TEST_RESULT_FV_LORA_AUG_20260723.json"

cd "$ROOT"

status() {
  if [[ ! -f "$WATCH_PID" ]]; then
    echo "WATCHER_STATUS=not_started"
    return 1
  fi
  local pid
  pid="$(<"$WATCH_PID")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "WATCHER_STATUS=running PID=${pid}"
  else
    echo "WATCHER_STATUS=not_running PID=${pid}"
  fi
  [[ -f "$STATE" ]] && cat "$STATE"
  tail -n 20 "$WATCH_LOG" 2>/dev/null || true
}

case "${1:-status}" in
  start)
    for path in "$WATCH_LOG" "$WATCH_PID" "$STATE" "$FREEZE" "$REPORT" "$FINAL"; do
      if [[ -e "$path" ]]; then
        echo "Refusing to reuse watcher artifact: ${path}" >&2
        exit 2
      fi
    done
    nohup env PYTHONDONTWRITEBYTECODE=1 "$PY" -u       scripts/watch_t1ce_fv_lora_then_test.py       --root "$ROOT"       --training-pid 3166       --training-output "$TRAIN_REL"       --training-log "${ROOT}/${TRAIN_REL}.launch.log"       --prereg workspace/BraTS/sammed3d_t1ce_wt/FV_LORA_AUG_PREREG_20260723.json       --test-manifest data/datasets/brats2024_random_subject_80_20_seed20260722_v1/sealed_test/test_cases.jsonl       --expected-test-manifest-sha256 7b91690c63a5eb359be105b9cc7c4ad563d41fca2b2c84e521251b7ed006b8d2       --evaluator scripts/evaluate_sammed3d_t1ce_wt_checkpoint.py       --freeze-output "$FREEZE"       --test-output "$REPORT"       --final-output "$FINAL"       --state-output "$STATE"       --poll-seconds 60       --device cuda       >"$WATCH_LOG" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$WATCH_PID"
    sleep 3
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Watcher exited during launch." >&2
      cat "$WATCH_LOG" >&2 || true
      exit 1
    fi
    echo "WATCHER_STARTED PID=${pid} STATE=${STATE}"
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 {start|status}" >&2
    exit 2
    ;;
esac
