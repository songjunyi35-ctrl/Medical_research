# Full972 保统计Gamma强度增强预注册

Status: frozen before split generation, implementation, GPU model construction, training, or new evaluation access.

## Objective and boundary

- Objective: 检验在现有增强后加入nnU-Net-style保均值/标准差Gamma，能否提高fresh泛化并超过Full972 development champion。
- Post-test development research; test270 outputs do not select any setting.
- Stage A only new train-only inner; Stage B gated fixed validation; test gated again.
- Source-inspired single augmentation, not full nnU-Net reproduction.

## Frozen inputs

- Full972 manifest972/444 SHA`6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Train cache index SHA`7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Native volume artifact SHA`32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Historical/EMA teacher SHA`f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`/`4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- Archived Full972 validation`0.7400125352044901/0.7312662169195357/103`,report SHA`30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44a66eba77d062d3ce129`.

## Split/seeds

- Fresh subject split seed20261080, exact108 inner,864 fit; freeze hashes before implementation.
- Stage A paired seed20261081; Gamma generator seed20261181.
- Stage B paired fresh seed20261082; Gamma generator seed20261182.
- Bootstrap Stage A/B seeds20261083/20261084.

## Base protocol

- Plain 4,700,897-param SegResNet,width16,dropout0.1,true-128³ image-only brain-crop T1ce→WT.
- Native-volume replacement,batch8,972 requested/968 consumed/121 steps.
- Random init,AdamW,LR2e-4 epochs1–100 then late-cosine denominator200,fixed stop168,clip5.
- Existing flip/shift8/scale0.9–1.1/offset±0.1/noise0–0.05.
- Dice1+CE0.1 plus voxel-KL T2 weight0.3 from0.75 history+0.25 EMA.
- Threshold0,argmax point,no TTA/SAM/postprocess,1× deployment.

## Frozen factor

- Control: existing augmentation only.
- Candidate: after existing augmentation, dedicated CUDA Generator; per sample probability0.3; non-inverted single-channel Gamma.
- Gamma sample: half uniform`[0.7,1]`,half uniform`[1,1.5]`; exact1 rejected from applied list.
- Transform min-max normalized values by power,gamma restore original range,then affine-match pre-Gamma per-sample mean and population std; epsilon1e-6.
- Gamma label untouched and dedicated RNG must leave global CPU/CUDA RNG unchanged.

## Stage A gate

- New864/108 zero-overlap split,paired seed20261081,fixed168,no inner loader during train.
- Exact initial state,all168 sampler SHA,first pre-Gamma existing-augmentation image/label hashes.
- Candidate applied fraction `[0.25,0.35]`,all gamma `[0.7,1.5]` and !=1,mean/population-std abs error<=1e-4,label exact,global RNG unchanged.
- Candidate strictly beats Control Scan/Subject,Point not lower,paired CI lower bounds>=-0.003;10k bootstrap seed20261083.
- Failure forbids Stage B.

## Stage B gate

- Gated paired full972 fresh seed20261082,fixed168,then one fixed-validation108 access.
- Candidate strictly beats matched Control and archived Full972 seed32 Scan/Subject,Point>=103 and not below Control,four CI lower bounds>=-0.003;10k seed20261084.
- All pass authorizes one frozen reused-test report and leaderboard updates; failure prohibits test.

## Stops/resources

- Any SHA/access/leakage/pairing/Gamma/RNG/label/reload/deploy/finite mismatch,teacher gradient,peak>24GiB,run>6h stops.
- No tuning probability/range/distribution/order/retain-stats/seed/epoch/LR/loss/threshold after access.
- No cache; planned workspace<300MB; each paired stage ~3.5–4h wall; inference remains1×.

## Sources

- https://www.nature.com/articles/s41592-020-01008-z
- https://github.com/MIC-DKFZ/nnUNet
- https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/color_transforms.py
