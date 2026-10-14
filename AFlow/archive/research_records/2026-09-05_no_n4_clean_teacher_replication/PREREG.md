# No-N4, fold-clean teachers: fixed preregistration

Status: frozen before split generation, candidate cache, model construction, and formal access.

## Inputs and roles

- Source: data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl; SHA256 6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65;972 scans/444 subjects, train only.
- Canonical cache workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json SHA256 7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd.
- Canonical label tensor SHA256 bf470d48a5f115f09407ad351bb6dee11ee36bdd0c2a4285ae0bd9cc22125724.
- Native-volume registry archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json SHA256 32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31; weights computed on fit IDs only.
- Exclusion inventory is frozen before splitting: all current data/datasets/*/*inner*cases.jsonl except paths containing smoke, plus original selector_cases.jsonl and sealed_confirmation/confirmation_cases.jsonl under brats2024_t1ce_plain_locator_confirm_seed20260817_v1. Existing failed-before-access v1 inner lists remain conservatively excluded. Expected eligible38 subjects/87 scans. Freeze every path/SHA and selected identities in SPLIT.json, no later discovery-driven seed search.
- Select E then F using existing deterministic shuffle/subset-sum routine with targets42 each, seeds20261301/20261302; F excludes E patients. Both must have at least12 subjects. If exact42/42 or expected pool unavailable, stop before training.
- Within each fold,930 fit is exact source complement; teacher and student fit identical and subject-disjoint from own inner. E/F inner patients mutually disjoint; cross-fold fit overlap expected. Fold labels and model metrics remain unopened until all completed/frozen.
- New cache is train-role only, same972 row order as canonical, exact same label bytes/crop geometry. Raw source values≤0 set0 in candidate to preserve the historical positive-support contract; no N4, no percentile/gamma/other transform. Subsequent existing preprocessing and resize functions unchanged.

## Seeds and fixed training

| Fold | history teacher | EMA teacher | paired students | bootstrap |
|---|---:|---:|---:|---:|
| E |20261311|20261312|20261313|20261331|
| F |20261321|20261322|20261323|20261332|

- Each teacher is supervised-only, original DiceCE (sigmoid, squared_pred=false, Dice1/CE0.1); no historical checkpoint or existing teacher is used.
- History teacher deploys raw epoch168. EMA teacher initializes an exact detached copy after epoch100, updates floating state0.999 old+0.001 current after each optimizer step101–168, copies integer buffers;8228 EMA updates. No raw/EMA comparison or choice.
- All teachers use canonical N4 cache and own fit930 only. Final teacher hash/state/fit provenance must freeze before their students. No teacher inner inference or teacher quality selection.
- Students canonical/no_n4 share own fold's two teachers with0.75/0.25 float32 raw-logit blend on each augmented input. Student loss=original DiceCE+0.3 binary voxel-KL, temperature2. Both teachers eval/no-grad.
- All models: plain AutomaticPromptLocator init_filters16 dropout0.1 center_aware=false,4,700,897 parameters; random init; AdamW weight_decay1e-5, default betas/eps, foreach/fused None; gradient clip5, bf16 autocast, no GradScaler.
- Batch8; fit-only native-volume weighted replacement requested972/consumed968/121 steps each epoch. Sampler private torch.Generator seeded by run seed. No workers RNG tuning. Historical flip/shift8/scale/offset/noise augmentation unchanged.
- LR2e-4 epochs1–100; from101 historical denominator99 cosine toward1e-5 at200; fixed stop168. No scheduler/epoch selection, warm start, checkpoint continuation, or early performance stopping.
- Each model max6h and24GiB peak allocated. NaN/Inf, invalid counts or provenance, teacher gradient, inconsistent hashes, access violation stop formal progression.
- Pairing: identical initial student state, sampling weights,168 sampler-index hashes, augmentation RNG-before/after, first augmented labels; input image hashes and first losses are expected to differ because preprocessing is the factor. Do not require false image/loss equality.

## Evaluation and gates

- Freeze4 student checkpoints plus4 teacher checkpoints/reports, source/split/cache/code/test/smoke/doc SHA256 in INNER_PREACCESS.json before any evaluator construction. Evaluator must reject incomplete, smoke, wrong-fold teacher provenance or wrong input cache.
- Every student runs its own fold42-case inner and own preprocessing cache exactly once. Both identity and all8 reflection logits produced in one fixed8-forward loop; identity first, raw logits inverse-flipped and float32 equal averaged.
- Axes exact:(),(-1,),(-2,),(-2,-1),(-3,),(-3,-1),(-3,-2),(-3,-2,-1).
- Raw logits>0; argmax raw logits point; no SAM, thresholds, connected components, morphology, calibration, model ensemble or view search. WT positive-case Point denominator, retain empty scans; Dice1 for both-empty else usual2intersection/(pred+target).
- Report42 scans and actual subjects per fold; Scan mean over scans; Subject mean of within-subject means. Absolute and paired CIs use10,000 patient-cluster resamples with seeds above, preserving all a sampled patient's scans. Same sampled clusters across compared arms. No scan-iid confidence claim.
- Per fold and per inference style(identity/Flip8): delta Scan>0,delta Subject>0,Point delta≥−1,paired Scan/Subject95% CI lower≥−0.003. Both folds and styles must pass.
- Cross-fold arithmetic average identity Scan delta>0.001 and Flip8 Scan delta>0.001; both mean Subject deltas>0. Engineering gates all pass. No retrospective subgroups as promotion gates.
- Any failed gate closes method without additional seeds/grid or old-cohort evaluations. No test270/UCSD75/fixed validation/outer/confirmation access. Formal inner outputs never guide a revision of this experiment.

## Conditional Full972

- Only if every Stage A gate passes: fit972 raw teacher seed20261341,EMA teacher20261342,paired students20261343,all fixed168 and identical recipes, now all972 fit including completed E/F inner roles. No inner/fixed-validation/test metric in final training.
- Only freeze both candidate/control checkpoints. Subsequent new external evaluation requires its own pre-label preregistration and source/format gates. Neither old leaderboard nor reused-cohort results can choose between fresh final models.
- Stage B only begins with≥8h remaining in this30h window; otherwise preserve positive Stage A and stage ready work without calling it completed final.

## Tests, smoke, archival

- Synthetic tests: deterministic42/42 selection and exclusions; fit/inner/teacher ownership; supported no-N4 processing preserves brain crop/label grid; EMA timing/count/update; model/loss constants; wrong-role/smoke/teacher rejection; paired cluster metrics/bootstrap; all gate directions.
- Real cache:972 labels exact canonical whole SHA and all crop geometry; canonical independent reconstruction on first3 train IDs. No-N4 independently reconstruct same3. Cache finite and same schema/order/count.
- ProportionalGPU smoke uses first8 fit cases only,2 optimizer steps per training role, teacher and paired student forward/backward finite, peak gate and reload exact. Synthetic EMA lifecycle handles100→101 boundary; short GPU smoke EMA checkpoint is explicitly not formal and cannot enter evaluation.
- New isolated entrypoints, historical source files unmodified. Record every formal failure and access boundary. Archive raw report/receipts and recompute hashes, update method catalog/current handoff. No namedtest is scheduled, so leaderboard metrics remain unchanged.
