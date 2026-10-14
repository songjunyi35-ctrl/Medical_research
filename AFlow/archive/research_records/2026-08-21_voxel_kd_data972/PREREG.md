# 972-case fixed-compute voxel-KD preregistration

Status: frozen before implementation, GPU smoke, formal training, and new fixed-validation access. Causal-control details are superseded by `AMENDMENT_001_MATCHED_CONTROL.md`.

## Hypothesis and causal contrast

The candidate repeats the successful historical168-to-200 voxel-KD protocol while changing only the eligible development-train population from 756 scans / 346 subjects to 972 scans / 444 subjects. Compute is fixed at 756 sampler draws per epoch, batch8 with drop_last, 94 optimizer steps per epoch, and 32 continuation epochs. Per Amendment001, a new cache-matched 756 arm is the causal control; the archived KD200 is a required secondary benchmark.

The 972 population is exactly the disjoint union of the historical 756 train, 108 selector, and 108 confirmation manifests. Those selector/confirmation roles are retired and are explicitly reclassified as development-train. The independent fixed validation108 is disjoint in scan and subject IDs. No test case enters training.

## Frozen evidence

- 972 manifest: `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl`, SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- 972 cache: `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train`, index SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`, `count=972`, `test_access=false`.
- Historical168 init: SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`.
- EMA168 teacher: SHA-256 `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- Archived 756-case KD validation: `workspace/BraTS/sammed3d_t1ce_wt/plain_voxel_kd_20260819/KD_VALIDATION.json`, SHA-256 `db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57`.
- Fixed validation manifest SHA-256 `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb`; implementation must bind and reject any mismatch.
- Fixed validation cache index canonical SHA-256 `9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322`.

## Frozen optimization

- Student: plain `AutomaticPromptLocator`, 4,700,897 parameters, historical168 initialization.
- Teachers: frozen historical168 and EMA168; teacher logits `0.75*history + 0.25*EMA`.
- Loss: MONAI `DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1)` plus `0.3 * binary_voxel_KL(T=2)` exactly as the archived runner.
- Candidate seed `20260846`; batch8; workers4; pin memory and persistent workers enabled; deterministic cuDNN; shift8 augmentation.
- `WeightedRandomSampler(candidate_weights, num_samples=756, replacement=True, generator=Generator(20260846))`; all 972 cases are eligible. Duplicates remain duplicates. `drop_last=True`, hence 752 consumed draws and 94 optimizer steps per epoch; the final 4 sampled indices are dropped.
- AdamW defaults with LR set by the archived epoch169-200 late-cosine function, base `2e-4`, minimum `1e-5`, weight decay `1e-5`; global clip norm5 after backward.
- Exactly 32 continuation epochs. Save global `176/184/192/200` only.
- Model/teacher construction finishes before resetting Python/NumPy/CPU/CUDA RNG to the frozen seed.
- BF16 autocast semantics remain identical to the archived voxel-KD runner: teacher and student forwards under autocast; teacher and student logits cast float32 for KL; DiceCE remains inside autocast.

## Technical gates before formal

1. Manifest union/disjointness, 756/346 and 972/444 arm counts pass.
2. Full overlapping 756 cases have image max-abs difference0 and labels exact between the 972 and current formal756 caches.
3. Control/candidate sampling weights contain756/972 entries; each sampler emits exactly756 valid indices and DataLoader consumes752 draws/94 batches.
4. Frozen loss helper equals the archived voxel-KD formula in value and logit gradient on asymmetric synthetic tensors.
5. Teacher parameters are frozen and receive no gradient; student loss/gradient are finite.
6. Real-cache GPU smoke uses batch8, full two-teacher/student forward, DiceCE+KL, backward, clip, step, and strict reload. Peak allocated must be at most24GiB and synchronized measured-step extrapolation at most45 minutes.
7. Implementation receipt freezes runner/evaluator/test hashes, evidence hashes, environment, access flags, and all gates before formal.

## Fixed validation and selection

- Cohort: canonical fixed validation108 / 50 subjects, raw-logit threshold `>0`, Point from argmax raw logit, no TTA/SAM/postprocessing.
- Evaluate only candidate epochs `176/184/192/200`.
- Select maximum candidate Scan Dice; exact tie by Subject Dice; final exact tie by earlier epoch.
- Compare selected candidate to the new same-global-epoch cache-matched `data756_control` and archived KD200.
- Paired percentile bootstrap: 10,000 draws with `numpy.random.default_rng`; Scan resamples 108 paired cases; Subject first averages scans within each of 50 subjects and then resamples subjects. Candidate-vs-same-epoch seeds `20261040..43`; candidate-vs-KD200 seeds `20261044..47`, indexed by selected/evaluated epoch order.
- Promotion requires candidate Scan and Subject strictly above both comparators, Point hits at least101/108, and all four Scan/Subject 95% CI lower bounds at least `-0.003`, plus every technical/provenance gate.
- No validation metric may change the loss, sampler, draw count, LR, epoch grid, threshold, checkpoint or postprocessing.

## Test governance

Only if every promotion gate passes, freeze the single selected checkpoint and final evaluator SHA in a pre-access receipt, then run one reused internal test270 evaluation. Otherwise `candidate_authorized_for_test=false`. Outer selector and confirmation are not available because their cases have been reclaimed into training. Test, outer, and confirmation access remain false during training and validation.
