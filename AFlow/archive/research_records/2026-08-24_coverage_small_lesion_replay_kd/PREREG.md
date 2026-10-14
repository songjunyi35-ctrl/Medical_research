# 覆盖优先的小病灶回放蒸馏预注册

Status: frozen before implementation, GPU access, training, and new validation access.

## Question and causal contrast

在972病例、752 effective draws/epoch和既有voxel-KD完全不变时，把原生体积加权带放回采样改成`680 coverage + 72 native-volume replay`，能否提高fixed validation108 Scan/Subject Dice且保持Point至少101/108？

Arms:

- `native_replacement_control`: 972-case native-volume weighted replacement.
- `coverage_replay_candidate`: 680 rolling-coverage draws + 72 consumed native-volume weighted replay draws + 4 dropped replay draws.

The sole arm contrast is the sampler policy. Both arms use 756 requested / 752 consumed draws and 94 optimizer steps per epoch.

## Frozen data roles

- Development-train: 972 scans / 444 subjects; manifest SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Fixed validation: 108 scans / 50 subjects; selection and promotion only.
- Outer selector and confirmation: previously revealed and reclassified into the 972 development-train pool; no longer independent evidence.
- Reused internal test270: no access unless every promotion gate passes; never pristine blind, external, or public-SOTA evidence.

## Frozen inputs

- Cache index SHA-256: `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Native-volume artifact SHA-256: `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Native972 sampling weight tensor SHA-256: `81d3fa6a907f791c4d0b97efe95ece5e057d15b85673463cadbeecc2b3f148ec`.
- Student/history teacher: `workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth`, SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`.
- EMA teacher: `workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/plain_128_full756_ema_fixed_epoch168.pth`, SHA-256 `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- Archived KD200 comparator: fixed validation report and checkpoint already recorded by the 2026-08-19 voxel-KD result.

No new cache is permitted.

## Frozen sampler

- Seed: `20260884`.
- Epoch request length: `756`; DataLoader batch8 with drop-last consumes `752` in `94` steps.
- Control: PyTorch `WeightedRandomSampler(weights, 756, replacement=True)` using a generator seeded with the frozen seed.
- Candidate coverage generator seed: `seed + 1001`.
- Candidate replay generator seed: `seed + 2003`.
- Candidate order-shuffle generator seed: `seed + 3007`.
- Candidate per epoch: take 680 indices from a persistent rolling queue; refill only when empty with a new `randperm(972)` from the coverage generator. Draw 76 replay indices with replacement from the frozen native weights. Deterministically shuffle 680 coverage + first72 replay indices; append the last4 replay indices so drop-last removes only replay.
- Candidate two-epoch coverage union: exactly all 972 population indices must appear at least once among coverage draws.
- Sampler output is process-local; two arms run in separate processes and do not use DistributedDataParallel.

The `680/72` effective split is fixed because 680 is 85 full batches and 72 is 9 full batches, giving a low replay share `9/94=9.57%` while preserving the historical 94-step budget. This ratio is not tunable after validation.

## Frozen model, optimization, and output

- Model: plain 4,700,897-parameter SegResNet locator, input `128^3` T1ce.
- Student init: historical epoch168.
- Teacher: `0.75 * history168 raw logits + 0.25 * EMA168 raw logits` on the same augmented image.
- Loss: Dice + `0.1` binary cross entropy + `0.3 * T^2 * voxel KL`, `T=2`.
- Seed `20260884`, batch8, workers4, shift8, AdamW historical defaults, base LR `2e-4`, minimum LR `1e-5`, weight decay `1e-5`, global epochs169–200, gradient clip5.
- Save only global epochs176/184/192/200.
- Output: raw logit `>0`; point is global argmax raw logit; no TTA, SAM, ensemble, calibration, or postprocessing.

## Selection, statistics, and gates

- Select candidate checkpoint only by fixed validation Scan Dice across 176/184/192/200; exact Subject tie, then earlier epoch tie.
- Compare selected candidate to same-epoch control and archived KD200.
- Required point estimates: candidate Scan and Subject strictly greater than both comparators; Point at least101/108.
- Safety statistics: scan- and subject-level paired percentile bootstrap, 10,000 draws with frozen evaluator seeds; all four 95% CI lower bounds at least`-0.003`.
- GPU smoke: four finite steps per arm, positive student gradients, no teacher gradients, peak allocated at most24GiB, mechanical formal extrapolation at most45min, strict reload max difference0.
- Formal:32 epochs,94 steps and752 consumed draws every epoch, four checkpoints, exact reload, peak allocated at most24GiB, elapsed at most45min per arm.
- Test authorization requires every implementation, smoke, formal, validation, point-estimate, CI, access, and hash gate. Otherwise `candidate_authorized_for_test=false`.

## Access declaration at freeze

- New fixed validation access for this method: false.
- Outer access: false.
- Confirmation access: false.
- Test270 access: false.
- Literature search used only public primary sources and did not expose project evaluation labels.
