#!/usr/bin/env bash
# Post-training readout for the gt_crop single-factor arm.
#
# Runs the independent strict-reload evaluation on the 108-case clean validation
# TWICE: once at the project's frozen 3-click protocol, once at SAM-Med3D's
# official 5-click default. Both are validation-only; the sealed 270-case test is
# NOT touched and requires a fresh operator decision.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aflow_env.sh"
ROOT="$AFLOW_REPO_ROOT"
PY="$AFLOW_PYTHON_RESOLVED"
cd "$ROOT"

ARM="workspace/BraTS/sammed3d_t1ce_wt/sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805"
CKPT="$ARM/sam_model_val_dice_best.pth"
CACHE="workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128/validation"
MANIFEST="data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl"
MSHA="fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c"

if [[ ! -f "$ARM/TRAIN_DONE.json" ]]; then
  echo "training not finished yet" >&2; exit 2
fi
CKPT_SHA=$(sha256sum "$CKPT" | cut -d' ' -f1)
echo "selected checkpoint SHA-256: $CKPT_SHA"

for NC in 3 5; do
  echo "=== validation, ${NC} clicks ==="
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=workspace/BraTS/env_shims:. "$PY" \
    scripts/evaluate_sammed3d_t1ce_wt_cached.py \
    --manifest "$MANIFEST" --expected-manifest-sha256 "$MSHA" \
    --expected-split validation \
    --cache-dir "$CACHE" \
    --checkpoint "$CKPT" --expected-checkpoint-sha256 "$CKPT_SHA" \
    --device cuda --seed 20260723 --num-clicks "$NC" \
    --output "$ARM/INDEPENDENT_VALIDATION_108_GTCROP_${NC}CLICK_20260805.json" \
    > "/tmp/claude-1000/-mlx-devbox-users-songjunyi-666-playground/523f6353-bc19-4376-aef6-b1a69c2dddcc/eval_gtcrop_${NC}.log" 2>&1
  echo "  -> $ARM/INDEPENDENT_VALIDATION_108_GTCROP_${NC}CLICK_20260805.json"
done

echo "=== summary ==="
"$PY" - <<'PYEOF'
import json
ARM = ("workspace/BraTS/sammed3d_t1ce_wt/"
       "sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805")
for nc in (3, 5):
    d = json.load(open(f"{ARM}/INDEPENDENT_VALIDATION_108_GTCROP_{nc}CLICK_20260805.json"))
    s = d["summary"]
    last = str(nc)
    print(f"\n{nc}-click  (cases={d['data']['selected_cases']}, "
          f"split={d['data']['expected_split']}, test_access={d['test_access']})")
    print(f"  geometry: {d['protocol']['spatial_preprocessing']}  "
          f"gt_localization={d['protocol'].get('uses_gt_for_localization')}")
    for k in sorted(s, key=int):
        print(f"    C{k}: scan={s[k]['scan_macro']['dsc_mean']:.6f}  "
              f"subj={s[k]['subject_balanced']['dsc_mean']:.6f}")
    print(f"  final-click scan   = {s[last]['scan_macro']['dsc_mean']:.6f}")
    print(f"  final-click subj   = {s[last]['subject_balanced']['dsc_mean']:.6f}")
PYEOF
