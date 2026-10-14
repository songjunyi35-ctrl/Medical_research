#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 {train|confirm}" >&2
  exit 2
fi
ACTION="$1"
case "$ACTION" in
  train|confirm) ;;
  *) echo "action must be train or confirm" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/aflow_env.sh"
cd "$REPO_ROOT"

CACHE="$REPO_ROOT/workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
SPLIT="$REPO_ROOT/data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1"
ROOT_OUT="$REPO_ROOT/workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817"
OUTPUT="$ROOT_OUT/plain_seed20260819"
CHECKPOINT="$OUTPUT/locator_val_dice_best.pth"
CONFIRMATION="$SPLIT/sealed_confirmation/confirmation_cases.jsonl"
REPORT="$ROOT_OUT/FINAL_CONFIRMATION.json"

require_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: $actual != $expected" >&2
    exit 2
  fi
}
require_sha d74ff2f0cf190c8a5458a70d19125d9a30603894d5f1799f192f0e60dd088fdb scripts/train_t1ce_wt_capg3d.py
require_sha 9e2bf5a8953574489a62db014627c33232555b3bcf2928b4ba00fc6a2980d1b5 scripts/evaluate_t1ce_wt_plain_locator_confirmation.py
require_sha 276ee7c131e413cf7943dc29e5372c1e5bbbb85c780e27f030343c5ef8bf4cd0 sammed3d/t1ce_wt_capg3d.py
require_sha a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16 "$SPLIT/train_cases.jsonl"
require_sha 3710d63842acaccb1e4fe81c5fade578b6f1a306363d969ea2d799edd6c32b07 "$SPLIT/selector_cases.jsonl"
require_sha dbf531992d1a39dcb70435068637c73687107bb9fa3f810be244966818f6113f "$CONFIRMATION"

"$AFLOW_PYTHON_RESOLVED" -c '
import json, sys
from pathlib import Path
index = json.loads((Path(sys.argv[1]) / "index.json").read_text())
expected = {
    "count": 972,
    "images_sha256": "f772520ca0d99fca0edc60aa04a7c73e0d7818eea9fa0174a29038f8334644b9",
    "labels_sha256": "bf470d48a5f115f09407ad351bb6dee11ee36bdd0c2a4285ae0bd9cc22125724",
    "split": "train",
    "test_access": False,
}
for key, value in expected.items():
    if index.get(key) != value:
        raise SystemExit(f"cache index mismatch for {key}: {index.get(key)!r} != {value!r}")
pre = index.get("preprocessing", {})
if pre.get("spatial") != "braincrop_resize" or pre.get("uses_gt_for_localization") is not False:
    raise SystemExit("cache is not image-only braincrop_resize")
' "$CACHE"

mkdir -p "$ROOT_OUT"
if [[ "$ACTION" == "train" ]]; then
  if [[ -e "$OUTPUT" ]]; then
    echo "refusing to overwrite training output: $OUTPUT" >&2
    exit 2
  fi
  env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1     "$AFLOW_PYTHON_RESOLVED" -u -m scripts.train_t1ce_wt_capg3d     --train-cache "$CACHE"     --validation-cache "$CACHE"     --train-case-ids "$SPLIT/train_cases.jsonl"     --validation-case-ids "$SPLIT/selector_cases.jsonl"     --train-cache-expected-split train     --validation-cache-expected-split train     --output-dir "$OUTPUT"     --arm plain     --device cuda:0     --seed 20260819     --epochs 100     --min-epochs 30     --patience 20     --batch-size 8     --validation-batch-size 8     --num-workers 4     --init-filters 16     --dropout 0.1     --lr 0.0002     --weight-decay 0.00001     --max-shift 8     --center-weight 0.1     --center-sigma 6     --ce-weight 0.1
  exit 0
fi

if [[ ! -f "$OUTPUT/TRAIN_DONE.json" || ! -f "$CHECKPOINT" ]]; then
  echo "completed training artifacts are missing" >&2
  exit 2
fi
if [[ -e "$REPORT" || -e "$REPORT.started.json" ]]; then
  echo "refusing to repeat or overwrite confirmation: $REPORT" >&2
  exit 2
fi
CHECKPOINT_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1   "$AFLOW_PYTHON_RESOLVED" -u -m scripts.evaluate_t1ce_wt_plain_locator_confirmation   --cache "$CACHE"   --confirmation-case-ids "$CONFIRMATION"   --locator-checkpoint "$CHECKPOINT"   --expected-checkpoint-sha256 "$CHECKPOINT_SHA"   --expected-confirmation-sha256 dbf531992d1a39dcb70435068637c73687107bb9fa3f810be244966818f6113f   --output "$REPORT"   --device cuda:0   --batch-size 8   --num-workers 4
