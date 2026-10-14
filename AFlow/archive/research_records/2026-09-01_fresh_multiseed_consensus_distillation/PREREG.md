# Fresh三seed共识蒸馏预注册

Status: frozen before split generation, implementation, GPU model construction, training, or new evaluation access.

## Objective and evidence boundary

- Objective: 独立验证三seed raw-logit共识是否能在fresh train-only inner同时改善Scan/Subject/Point，并将该共识蒸馏进1×student。
- This is post-test research, but method choice/weights/epochs/gates do not use GCSAM/test270 per-case outputs.
- Fixed validation108 and reused test270 are not accessed during Stages A/B. Stage C validation requires both earlier stages to pass. Test270 is not authorized.
- This is source-inspired output-level ensemble distillation, not a literal reproduction of the cited 2D retinal or multi-population brain-tumor papers.

## Frozen base inputs

- Full972 source manifest: `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl`, 972 scans / 444 subjects, SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Train cache index: `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json`, SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Native WT-volume artifact: `archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json`, SHA-256 `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Historical teacher checkpoint SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`.
- EMA168 teacher checkpoint SHA-256 `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- Stage C Full972 teachers:
  - seed20260932 SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`.
  - seed20260940 SHA-256 `48c366ef9f5d3ba5c667482deda6df90fa7cd4bf03c314017e15c13f161dfd38`.
  - seed20260941 SHA-256 `00db7860b7951d1f4b0affde6cad558f679421ba93dfe69157c31bd497b19ddb`.
- Archived Full972 seed32 fixed-validation comparator: Scan/Subject/Point `0.7400125352044901/0.7312662169195357/103/108`; raw validation report SHA-256 `30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44e66eba77d062d3ce129`.

## Fresh split and seeds

- Split algorithm: `seeded_subject_shuffle_exact_case_subset_sum`, seed`20261030`, exact108 train-only inner scans; remaining864 fit scans. Exact case/subject counts and role SHA values must be frozen before implementation.
- Fresh fit864 teacher seeds: `20261031`, `20261032`, `20261033`.
- Paired fit864 student seed: `20261034`.
- Paired Full972 final student seed: `20261035`.
- No seed may be dropped or replaced after any inner/validation access.

## Frozen architecture/training protocol

- Model: plain `AutomaticPromptLocator` / MONAI SegResNet, `init_filters=16`, dropout`0.1`, 4,700,897 parameters.
- Input/target: true-128³ image-only brain-crop single-channel T1ce → binary WT labels `{1,2,3}`.
- Sampling: native-volume weighted replacement; batch8; 972 requested / 968 consumed / 121 optimizer steps each epoch.
- Schedule: random initialization; global epochs1–100 LR`2e-4`; epochs101–200 cosine denominator; fixed stop at epoch168; AdamW weight decay`1e-5`; shift8; gradient clip5.
- Supervision: sigmoid DiceCE with Dice weight1 and CE weight0.1.
- KD: binary voxel KL, temperature`2`, weight`0.3`, including the standard`T²` factor.
- Output: raw logit threshold`>0`; point=raw-logit argmax; no TTA, SAM, connected-components, threshold search, or postprocessing.

## Stage A: fresh teachers and consensus

- Train all three teachers on fit864 to fixed epoch168 using the frozen historical/EMA raw-logit teacher `0.75/0.25`; only seed changes.
- No inner loader/path is constructed during teacher training.
- After all three reports/checkpoints pass, evaluate each teacher and fixed raw-logit consensus `(z31+z32+z33)/3` once on fresh inner108.
- Consensus must strictly exceed all three teachers on Scan and Subject, Point must be at least the maximum individual Point, and all six paired Scan/Subject CI lower bounds must be`>=-0.003` (10,000 draws; bootstrap seed`20261036`).
- Failure sets `student_stage_authorized=false`.

## Stage B: compute-matched students

- Only Stage A pass authorizes Stage B.
- Both arms use exact same student initial state, fit864 cases, epoch index sequences, augmentations, optimizer/LR, losses, and epoch168 stop.
- Control performs three real no-grad forwards of teacher seed20261031 and averages identical raw logits.
- Candidate performs one no-grad forward from each teacher seed20261031/32/33 and averages raw logits equally.
- All teachers are eval/frozen, receive identical augmented input, and must have no gradient.
- Candidate must strictly exceed Control Scan and Subject, Point must not decrease, and paired Scan/Subject CI lower bounds must be`>=-0.003` (10,000 draws; seed`20261037`).
- Initial state, all epoch sampler-index SHA values, first augmented batch, teacher-forward count3, first student loss for control/candidate at identical teacher logits parity preflight, reload and deploy outputs are frozen engineering gates.
- Failure sets `full972_final_authorized=false`.

## Stage C: Full972 final and fixed validation

- Only Stage B pass authorizes Stage C.
- Both arms train full972 from exact same fresh random state to epoch168.
- Control repeats Full972 seed20260932 teacher three times; Candidate uses Full972 seeds20260932/40/41 once each.
- After training and pre-access receipt, evaluate both once on fixed validation108 with threshold0 and identical output contract.
- Candidate must strictly exceed same-epoch matched Control and archived Full972 seed32 on Scan/Subject, Point`>=103/108`, and four paired Scan/Subject CI lower bounds`>=-0.003` (10,000 draws; seed`20261038`).
- Stage C pass produces a test candidate only. Reused test270 requires a new receipt and explicit user authorization.

## Engineering stops and resource bounds

- SHA/access/split leakage/parameter-count/sampler/teacher-count/initial-state/finite-gradient/reload/deploy mismatch stops the affected stage.
- Teacher gradient, NaN/Inf, batch8 peak allocated`>24GiB`, or any single arm elapsed`>6h` stops the affected stage.
- No formal stage may create a multi-GiB cache. Planned addition`<300MB`; disk must be checked before formal.
- Failed gates prohibit changing teacher membership, `1/3` weights, seed, KD temperature/weight, epoch, LR, threshold, TTA, SAM, or postprocessing.

## Sources

- https://arxiv.org/abs/2509.11689
- https://www.nature.com/articles/s41598-026-35627-x
- https://github.com/AhmeddEmad7/Brain-Tumor-Segmentation-Advancing-Generalizability
- https://www.nature.com/articles/s44401-026-00108-w

