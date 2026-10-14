# Full972 LAWA6同轨迹稳定性预注册

Status: frozen before split generation, implementation, GPU model construction, training, or new evaluation access.

## Objective and evidence boundary

- Objective: 检验同一Full972-style轨迹最近六个逐epoch checkpoint的等权参数平均，能否跨两个fresh inner seed稳定超过各自raw epoch168，并在第三个fresh Full972 seed的fixed validation上超过matched raw与archived development champion。
- This is post-test development research. GCSAM/test270 per-case outputs are not read or used to choose `k`, epochs, coefficients, seed, threshold, or gates.
- Stage A只允许全新train-only inner；Stage B只有Stage A全过门才允许fixed validation108。Reused test270只有Stage B全过门才允许一次固定报告，不用于选择。
- This is a source-inspired 3D AdamW/late-cosine LAWA adaptation, not a literal reproduction of ImageNet/BERT/CIFAR studies.

## Frozen base inputs

- Full972 source manifest: `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl`, 972 scans / 444 subjects, SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Train cache index: `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json`, SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Native WT-volume artifact: `archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json`, SHA-256 `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Historical teacher SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`.
- EMA168 teacher SHA-256 `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- Archived Full972 seed32 checkpoint SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`.
- Archived Full972 fixed-validation comparator: Scan/Subject/Point `0.7400125352044901/0.7312662169195357/103/108`; raw report SHA-256 `30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44e66eba77d062d3ce129`.

## Fresh split and seeds

- Split algorithm: `seeded_subject_shuffle_exact_case_subset_sum`, seed`20261060`, exact108 train-only inner scans; remaining864 fit scans. Exact role counts and SHA values must be frozen before implementation.
- Stage A fit864 seeds: `20261061`, `20261062`; both fixed epoch168, no inner loader during training.
- Stage B Full972 seed: `20261063`; only Stage A all-pass authorizes training.
- Bootstrap seeds: Stage A `20261064`; Stage B `20261065`.
- No seed may be replaced after any inner/validation access.

## Frozen architecture and training protocol

- Model: plain `AutomaticPromptLocator` / MONAI SegResNet, `init_filters=16`, dropout`0.1`, 4,700,897 parameters.
- Input/target: true-128³ image-only brain-crop single-channel T1ce → binary WT labels `{1,2,3}`.
- Sampling: native-volume weighted replacement; batch8; 972 requested / 968 consumed / 121 optimizer steps per epoch.
- Schedule: random initialization; epochs1–100 LR`2e-4`; epochs101–200 late-cosine denominator; fixed stop epoch168; AdamW weight decay`1e-5`; shift8; gradient clip5.
- Supervision: sigmoid DiceCE, Dice weight1 and CE weight0.1.
- KD: historical/EMA raw-logit teacher`0.75/0.25`; binary voxel KL, temperature`2`, weight`0.3`, including`T²`.
- Output: raw logit threshold`>0`; point=raw-logit argmax; no TTA, SAM, connected-components, threshold search, or postprocessing.

## Frozen LAWA6 factor

- Capture the student state after epochs`163,164,165,166,167,168` only.
- Candidate state is the uniform arithmetic mean with coefficient`1/6` for every floating tensor.
- Non-floating tensors must be exact across all six captures and are copied unchanged; otherwise stop.
- Accumulation must be deterministic in CPU float64, division occurs once after the sixth capture, and values are cast back to each original dtype for the candidate checkpoint.
- Raw control is the exact epoch168 student state from the same run. The optimizer, RNG, teacher forwards, batches, updates, and runtime path are shared; LAWA must not feed back into training.
- Both raw and LAWA checkpoint reloads and direct-state vs checkpoint logits must be exact within dtype-aware pre-registered tolerances; state-key/shape sets must be exact.

## Stage A: two-seed fresh inner gate

- Train both fit864 seeds independently and in parallel on separate GPUs. Training constructs no inner/fixed-validation/test loader.
- After both `TRAIN_DONE` reports, raw/LAWA checkpoints, hashes, capture epochs, finite/resource/reload gates pass, write a pre-access receipt and evaluate all four models once on the frozen fresh inner108.
- For each seed, LAWA must strictly exceed raw on Scan and Subject, Point must not decrease, and paired Scan/Subject 95% CI lower bounds must each be`>=-0.003`.
- CI uses the repository's established 10,000-draw paired bootstrap with seed`20261064`; report improved/worsened/tied scan and subject counts.
- All eight gates across two seeds are conjunctive. Any failure sets `full972_final_authorized=false`.

## Stage B: Full972 final and fixed validation

- Only Stage A all-pass authorizes a seed`20261063` Full972 run to fixed epoch168 with the identical LAWA6 contract.
- After training, freeze raw/LAWA checkpoints and a pre-access receipt, then evaluate both once on fixed validation108.
- LAWA must strictly exceed matched raw and archived Full972 seed32 on Scan and Subject; Point must be at least matched raw and at least`103/108`.
- Four paired Scan/Subject 95% CI lower bounds relative to the two comparators must all be`>=-0.003`; 10,000 draws, seed`20261065`.
- Stage B all-pass authorizes one frozen reused test270 report under the user's explicit leaderboard objective. The test report must compare against both GCSAM and Full972 KD, cannot tune any choice, and triggers both leaderboard updates.

## Engineering stops and resource bounds

- SHA/access/split leakage/parameter-count/sampler/capture-epoch/state-key/non-floating-buffer/finite-gradient/reload/deploy mismatch stops the affected stage.
- Frozen teachers must receive no gradient. LAWA capture/average must not change training RNG or raw epoch168 state.
- Batch8 peak allocated`>24GiB` or any single run elapsed`>6h` stops the affected stage.
- No multi-GiB cache. Planned addition`<500MB`; disk must be checked before formal.
- Failed gates prohibit changing `k=6`, epoch window, coefficients, seed, KD temperature/weight, LR, threshold, TTA, SAM, or postprocessing.

## Sources

- https://arxiv.org/abs/2209.14981
- https://github.com/JeanKaddour/LAWA
- https://arxiv.org/abs/1803.05407
- https://docs.pytorch.org/docs/stable/optim.html#stochastic-weight-averaging
