# 972病例配额一致voxel-KD预注册

Status: frozen before implementation, GPU smoke, formal, and new validation access.

## Causal question

在972病例池、原生体积权重和voxel-KD完全相同时，把每轮有效draw从752提高到968，能否超过same-run control和archived KD200？唯一arm差异是draw/optimizer-step预算。

## Frozen inputs

- Train:972 scans / 444 subjects，manifest SHA `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`。
- Train cache index SHA `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`。
- Native-volume artifact SHA `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`；native972 weight SHA `81d3fa6a907f791c4d0b97efe95ece5e057d15b85673463cadbeecc2b3f148ec`。
- Historical168 init SHA `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`。
- EMA168 teacher SHA `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`。
- Fixed validation108/50 manifest/index SHA `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb` / `9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322`。
- Archived KD validation SHA `db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57`。

## Frozen arms

- `budget752_control`: `WeightedRandomSampler(native_weights,756,replacement=True,Generator(20260924))`; batch8/drop-last→752 draws,94 steps/epoch。
- `budget968_candidate`: `WeightedRandomSampler(native_weights,972,replacement=True,Generator(20260924))`; batch8/drop-last→968 draws,121 steps/epoch。
- Both:32 epochs global169–200；control3008 total steps，candidate3872 total steps。

## Frozen model and optimization

Same 4,700,897-parameter locator, history168 init, `0.75/0.25` history/EMA teacher, Dice+0.1 BCE+0.3 voxel-KL at T2, seed20260924, batch8, workers4, shift8, AdamW historical defaults, base LR2e-4/min1e-5, wd1e-5, clip5, checkpoints176/184/192/200, threshold0, no TTA/SAM/postprocess.

## Selection and promotion

- Select candidate by validation108 Scan, Subject exact tie, earlier epoch exact tie.
- Candidate must strictly beat same-epoch control and archived KD200 Scan/Subject; Point>=101/108。
- Four paired 10,000-draw percentile bootstrap CI lower bounds >=-0.003。
- Engineering gates: exact counts/hashes/reload；peak<=24GiB；control<=45min；candidate<=60min。
- All gates authorize one reused internal test270; otherwise stop and archive without tuning。

## Access at freeze

validation=false; outer=false; confirmation=false; test270=false。Outer/confirmation病例已回收为train，不再是独立证据。
