# LoMix-3D GPU smoke cost update

This is an execution-time cost update, not a scientific protocol change.

- Frozen method, data, batch, loss, epoch, selection, promotion, threshold, and access rules are unchanged.
- Real train-cache batch: `8`; optimizer steps: `1`; all smoke gates passed.
- RNG-safe v2 step wall time: `1.474607s`.
- Mechanical `94 steps/epoch × 32 epochs` extrapolation: `4435.62s`, about `73.9min`.
- Peak allocated/reserved GPU memory: `25,190,449,152 / 31,461,474,304` bytes (`23.46 / 29.30 GiB`).
- This replaces the pre-smoke `15–35min` estimate for planning only. Formal elapsed time remains to be measured.
- The first `2.887293s` smoke used trainer SHA `1ed3bd...`; review then found that deploy-equivalence audits could consume the future dropout RNG stream. No formal training/validation/test had run. That result is retained under `smoke_seed20260846/` but superseded by the RNG-preserving v2 implementation and smoke.
- No validation, outer, confirmation, or test data were accessed.

Current evidence: `workspace/BraTS/sammed3d_t1ce_wt/plain_lomix3d_20260819/smoke_v2_rngsafe_seed20260846/GPU_SMOKE.json`, SHA-256 `4df452406b251f4901fe7e6f139568fb5ace059fef082b8b11365b5b499e1b18`.
