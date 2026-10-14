# Preregistration implementation amendment 1

Frozen before any GPU smoke or formal cache/training result.

The first 12-case 128³ cache build completed, but the builder's post-build image byte check failed because cache workers explicitly set SimpleITK to one thread while the verification process inherited the host thread-pool default. N4 floating reductions therefore did not follow an identical execution path. No model training or inner-tune evaluation occurred.

The preprocessing and training protocol are unchanged. Raw reproduction is moved to `scripts/verify_t1ce_wt_braincrop_cache_variable_resolution.py`, which explicitly fixes SimpleITK to one thread before rerunning the same builder `_prepare` function. Every formal arm must pass exact image bytes, exact label bytes, and exact native WT-volume reproduction on four preregistered evenly spaced cache slots. Formal builder invocations use `--verify 0` to disable the mismatched in-process checker, then immediately run the dedicated verifier.
