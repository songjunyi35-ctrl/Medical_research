# Plain locator 128³ → 160³ strict single-factor preregistration

Status: frozen before cache build smoke and formal training on 2026-08-18 UTC.

This is a development-only experiment inside the same historical 756-case train cohort. A new seed `20260828` produced a subject-clean 648-case/294-subject fit view and 108-case/52-subject inner-tune view, with zero case and subject overlap. It is not external or pristine validation.

The only factor is the cubic preprocessing/model grid: a newly built true-128³ control versus a newly built true-160³ candidate. Both caches start from the authorized raw NIfTI cases and use the same ordered chain: canonical RAS → N4 shrink-2 → image-only nonzero bounding-box crop → source nonzero-population z-score → trilinear image / nearest-exact label resize. Neither arm may use an existing 128³ cache as preprocessing input.

Both arms use training seed `20260829`, native-grid WT volume-based sampling weights, batch 8, 4,700,897 parameters, DiceCE with CE weight 0.1, AdamW at `2e-4` with weight decay `1e-5`, literal `max_shift=8`, and raw logit `>0`. They stop at fixed epoch168 on the same 200-epoch-defined schedule: epochs 1–100 constant, followed by the locked late-cosine function toward `1e-5`. Inner tune is not read until the final optimizer step, so it cannot select an epoch.

All gates must pass: scan and subject Dice gains each at least `+0.005`; paired subject bootstrap 95% CI lower bound at least zero (10,000 samples, seed `20260830`); point-hit decline no worse than `-0.02`; exact checkpoint reload difference zero. Failure makes 160³ negative with no scale, epoch, threshold, or postprocessing tuning. Passing records only an inner-positive development result and does not itself authorize any later cohort access.

CAPG, grayscale residual, TTA, postprocessing, and SAM remain excluded. Outer selector, confirmation, and the 270-case test remain unaccessed.

The complete machine contract, hashes, and access flags are in `PREREG.json`.
