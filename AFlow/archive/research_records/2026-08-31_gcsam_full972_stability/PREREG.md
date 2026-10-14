# Full972 GCSAM跨seed稳定性预注册

Status: frozen before split generation, implementation, GPU model construction, training, or new evaluation access.

## Scientific question and arms

- Question: 在Full972、相同teacher/loss/model/data下，2025 GCSAM是否能用更平坦且低噪声的更新轨迹，在train-only inner和fresh-seed final两个阶段都超过AdamW。
- `adamw_control`: 历史AdamW，一次student forward/backward，clip5后step。
- `gcsam_candidate`: paper-form GCSAM，`rho=0.05`、non-adaptive；第一次梯度对所有`ndim>1`参数按dims`1..N-1`中心化，以中心化梯度全局L2 norm归一化扰动；同一batch/augmentation/teacher logits第二次forward/backward，clip5，restore，再由同一AdamW step。
- 唯一arm差异是student optimizer update；teacher、loss与部署输出不变。

## Frozen inputs

- Full972 manifest: `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl`, 972 scans / 444 subjects, SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Train cache index: `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json`, SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Native WT volume artifact: `archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json`, SHA-256 `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Historical teacher: `plain_fixed_epoch168.pth`, SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`.
- EMA teacher: `plain_128_full756_ema_fixed_epoch168.pth`, SHA-256 `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- Literature implementation boundary: paper Eq. `epsilon_GC=rho*g_GC/||g_GC||` takes priority over the official example's pre-centralization norm ordering. No implementation comparison is permitted after any inner/validation access.

## Split and seeds

- Generate one new train-only subject-disjoint split from Full972 with algorithm `seeded_subject_shuffle_exact_case_subset_sum`, seed `20261001`, exact inner tune 108 scans; expected fit 864 scans. No fixed validation, outer, confirmation, or test path may be read by the split builder.
- Inner paired seed: `20261002`; both arms must have exact initial state SHA and exact per-epoch sampled index SHA.
- Final paired fresh seed: `20261003`; both arms train full972 from exact same initial state to the inner-selected epoch.
- Batch8; 972 requested / 968 consumed / 121 optimizer steps per epoch; weighted replacement from frozen native-volume weights.

## Frozen model, teacher, loss, and schedule

- Plain `AutomaticPromptLocator` / SegResNet, `init_filters=16`, dropout`0.1`, 4,700,897 parameters.
- Teacher raw logit: `0.75*history168 + 0.25*EMA168`, no teacher gradient.
- Loss: MONAI sigmoid DiceCE with `lambda_dice=1`, `lambda_ce=0.1`, plus binary voxel KL weight`0.3`, temperature`2`.
- AdamW base LR`2e-4`, weight decay`1e-5`; epochs1–100 constant LR, epochs101–200 cosine to`1e-5`; max shift8; gradient clip5 on the update gradient.
- Inner runs exactly200 epochs. Candidate selects only epoch101–200 by `(Scan, Subject, -epoch)`; control is compared at the candidate-selected epoch.
- Final stop epoch equals the exact inner-selected epoch and uses the same 200-epoch LR denominator.
- Output: raw logit threshold`>0`; point=argmax raw logit; no Segment Anything Model, test-time augmentation, connected components, postprocessing, or threshold search.

## Engineering gates before formal

- Synthetic tests must establish: centralization leaves1D gradients unchanged; every`ndim>1` gradient has near-zero mean over dims`1..N-1`; global perturbation norm equals`0.05` within tolerance; `first_step→restore` reproduces every parameter exactly; second-step AdamW equals a hand-computed reference; zero/nonfinite gradients fail safely; teacher gradients remain absent.
- `rho=0` parity must match control loss, gradient after clip, and updated parameters within frozen numerical tolerance on CPU and CUDA BF16 smoke.
- Paired smoke must establish exact input/label/index hash, exact initial model SHA, finite two-pass loss, checkpoint reload equality, output schema, and deployment-logit equality.
- Peak allocated hard limit24GiB per arm. Control formal hard limit6h; candidate inner9h/final8h.
- Disk must be checked before formal; no new cache and planned experiment addition<300MB.

## Promotion and stop gates

- Inner candidate must strictly beat same-epoch control Scan and Subject; Point may be at most1/108 lower; paired Scan and subject-level Subject 10,000-bootstrap 95% CI lower bounds must both be`>=-0.003`.
- Inner failure stops the experiment before full972 final and before fixed validation.
- Final candidate must strictly beat matched control and archived Full972 seed32 Scan/Subject `0.740012535/0.731266217`; Point`>=103/108`; the four paired Scan/Subject CI lower bounds versus both comparators must be`>=-0.003`.
- Any hash/pairing/restore/reload/finite/memory/time/access gate failure stops the affected stage.
- No post-result changes to rho, gradient dimensions, adaptive flag, clip, optimizer hyperparameters, epoch, seed, threshold, teacher, loss, or postprocessing.

## Evidence access and reporting

- At freeze: new train-only inner=false; new fixed validation=false; outer=false; confirmation=false; new test=false.
- Existing validation/test summary numbers may be used only as archived comparators. No sample tensor or per-case result is accessed before the authorized stage.
- Outer selector and confirmation are development-train roles and will not be accessed or relabeled as evaluation.
- Even if all final gates pass, reused test270 requires a new frozen pre-access receipt and explicit user authorization. It is never pristine blind/external evidence.
- After each stage, update `RESULT.md`/machine JSON, current handoff, and method catalog. If test is ever run, update both leaderboard files and validate them.

## Sources

- GCSAM 2025 paper: https://arxiv.org/abs/2501.11584
- IEEE Access DOI: https://doi.org/10.1109/ACCESS.2025.3624332
- Official code: https://github.com/mhassann22/GCSAM
