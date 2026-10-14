# Amendment 002: Stage B Full972 implementation and smoke freeze

Status: frozen before Stage B formal model construction or optimizer step and before fixed-validation/outer/confirmation/test access.

This amendment does not change the method, Full972 role, seed20261063, epoch168, `k=6`, capture epochs163–168, uniform coefficients, training protocol, metrics, gates, threshold, or access rules.

## Authorization

- Stage A decision SHA-256: `2437a641c28c218878e5df45c784b2b3154718416d0dd920a5ca10b10ad59001`.
- Stage A result: 10/10 promotion gates passed; `full972_final_authorized=true`.
- Stage A access at authorization: train-only inner accessed; fixed validation108, outer, confirmation, and test270 not accessed.

## New entry point

- Stage B wrapper: `scripts/train_t1ce_wt_lawa6_stage_b_final.py`, SHA-256 `be33d68018cad32bfd940609d86daa654b49fd461fa91e5e82a451dd1436e5db`.
- Launcher: `scripts/run_t1ce_wt_lawa6_stage_b_final.sh`, SHA-256 `9865875cc821ecf5c0f0354870d678b06043bc17901e61a95b2158a280eab0bb`.
- Wrapper reuses the already smoke/formal-proven Stage A trainer without modifying that historical file, then specializes only the pre-registered role constants: Full972 manifest SHA,972 scans/444 subjects, seed20261063, Stage A decision SHA, and Stage B schema.
- Wrapper validates the exact Stage A decision hash, schema, 10/10 gates, authorization boolean, and access seal before delegating to the frozen trainer.

## Pre-formal test process

- Initial combined test run: Stage B specialization test passed, then four Stage A gate tests failed because the test itself left the imported Stage A module constants mutated in the shared pytest process. No GPU model, formal training, fixed validation, outer, confirmation, or test access occurred.
- Fix boundary: only `tests/test_lawa6_stage_b_final.py` was changed to restore all imported module constants in a `finally` block after asserting specialization. No production code, method factor, data, seed, training or gate changed.
- Fixed test SHA-256: `cd805954a2c5b2d579f9f7933ad4e48058830eb9c0a3efbee0b3e80cde94a871`.
- Final combined result: `11 passed` for Stage B authorization/specialization, Stage A promotion gates, and LAWA accumulation contracts.

## Stage B GPU smoke

- Smoke report SHA-256: `6cc5b5adfd98c78a4fb47c3297b8b1212caa6422e5fb698b1e0b81f5cd95ec02`.
- Raw/Lawa smoke checkpoint SHA-256: `487c427af6b58d132adc8b46f6311886c8b95556edc104661ce52b0c4aac9db8` / `b17b0a90fb9798b5f4dd61b2578b84c0d778c4b5c1d9cba06e2c7f1f81d3715e`.
- Six captures, CPU FP64 averaging, raw/LAWA reload, deploy-logit exactness, finite loss/gradient, nonzero parameter delta, access seal and 24GiB gates all passed.
- Peak allocated: `21,796,646,912` bytes (about20.30GiB).
- Smoke LAWA-vs-raw relative L2: `0.0106870241`.
- Available disk immediately before formal: about15G; no new cache; planned formal addition about120MB.

## Access at freeze time

- Train-only inner: accessed once in Stage A.
- Fixed validation108: not accessed.
- Outer: not accessed.
- Confirmation: not accessed.
- Test270: not accessed.
