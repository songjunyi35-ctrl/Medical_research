# Full972 Lookahead5稳定优化预注册

Status: frozen before split generation, implementation, GPU model construction, training, or new evaluation access.

## Objective and evidence boundary

- Objective: 检验从随机初始化开始贯穿全轨迹的Lookahead慢权重，能否比完全配对AdamW更稳定，并在fresh Full972 final超过archived development champion。
- This is post-test development research. No test270 per-case output is used to choose method,`k`,alpha,seeds,epoch,threshold,or gates.
- Stage A只允许新train-only inner；Stage B只有Stage A全过门才允许fixed validation108；test只有Stage B全过门才允许一次固定报告。
- This is a source-inspired Lookahead-over-AdamW 3D adaptation, not a literal reproduction of the paper.

## Frozen base inputs

- Full972 source manifest: `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl`,972 scans/444 subjects,SHA-256`6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Train cache index SHA-256`7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Native WT-volume artifact SHA-256`32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Historical teacher SHA-256`f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`.
- EMA168 teacher SHA-256`4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- Archived Full972 seed32 fixed-validation comparator`0.7400125352044901/0.7312662169195357/103/108`; report SHA-256`30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44a66eba77d062d3ce129`.

## Fresh split and seeds

- Split algorithm:`seeded_subject_shuffle_exact_case_subset_sum`, seed20261070, exact108 train-only inner scans; remaining864 fit. Exact role counts/hashes freeze before implementation.
- Stage A paired arms share seed20261071 and exact random initialization.
- Stage B paired arms share fresh seed20261072; only Stage A all-pass authorizes it.
- Bootstrap seeds: Stage A20261073; Stage B20261074.
- No seed may be changed after any inner/validation access.

## Frozen architecture/training protocol

- Plain AutomaticPromptLocator/MONAI SegResNet,init_filters16,dropout0.1,4,700,897 parameters.
- True-128³ image-only brain-crop T1ce→binary WT`{1,2,3}`.
- Native-volume weighted replacement,batch8,972 requested/968 consumed/121 steps per epoch.
- Random initialization; epochs1–100 LR`2e-4`; epochs101–200 late-cosine denominator; fixed stop168; AdamW weight decay`1e-5`; shift8; clip5.
- DiceCE(Dice1,CE0.1)+binary voxel-KL temperature2 weight0.3 from`0.75 history168+0.25 EMA168`raw logits.
- Raw logit threshold`>0`,point=argmax,no TTA/SAM/postprocessing.

## Frozen Lookahead factor

- Control: unmodified AdamW.
- Candidate: same AdamW as fast optimizer; slow parameters initialized as exact CPU/GPU clone of step0 trainable parameters.
- After each fast step increment counter. Every`k=5` steps, for each trainable floating parameter execute`slow.add_(fast-slow, alpha=0.5)`then`fast.copy_(slow)` under no-grad.
- Total formal steps=`168×121=20328`; expected slow updates=`floor(20328/5)=4065` with3 trailing fast steps. Immediately after final fast step, copy the latest slow weights to model for checkpoint/evaluation without another interpolation.
- Candidate checkpoint must state slow-weight deployment. Control checkpoint remains exact AdamW step20328.
- Lookahead operations must not consume RNG. No extra forward/backward or change to AdamW moments/LR/gradient clipping.

## Stage A gate

- New subject-clean split:864 fit/108 inner, zero case/subject overlap.
- Train paired Control/Candidate to fixed epoch168 without constructing inner loader.
- Before inner access require both TRAIN_DONE reports, checkpoint/reload/deploy/finite/resource gates, exact initial state, exact168 sampler-index hashes, exact first augmented image/label hashes, candidate20328 fast steps/4065 slow updates/final-slow deployment, and a preaccess receipt.
- Candidate must strictly exceed Control Scan/Subject,Point must not decrease,paired Scan/Subject CI lower bounds must each be`>=-0.003`;10,000 bootstrap draws,seed20261073.
- Failure sets`full972_final_authorized=false`.

## Stage B gate

- Only Stage A all-pass authorizes paired full972 Control/Candidate with fresh shared seed20261072 and fixed epoch168.
- After training and preaccess receipt, evaluate both once on fixed validation108.
- Candidate must strictly exceed matched Control and archived Full972 seed32 on Scan/Subject,Point>=103 and not below Control; four paired CI lower bounds`>=-0.003`;10,000 draws,seed20261074.
- All-pass authorizes one frozen reused-test270 report and mandatory leaderboard updates. Failure prohibits test.

## Engineering/resource stops

- Any SHA/access/leakage/pairing/step/slow-update/RNG/reload/deploy/finite mismatch stops the stage.
- Teacher gradient,peak>24GiB,or any run>6h stops the stage.
- No multi-GiB cache; planned addition<300MB; check disk before formal.
- Failed gates prohibit changing`k=5`,alpha0.5,slow/fast choice,seed,epoch,LR,loss,threshold,TTA,SAM,or postprocessing.

## Sources

- https://arxiv.org/abs/1907.08610
- https://papers.neurips.cc/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html
- https://github.com/michaelrzhang/lookahead
