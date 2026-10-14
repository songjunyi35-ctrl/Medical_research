# AFlow BraTS 科研交接

> **Superseded on 2026-08-10.** 本文件保留完整历史，不再提供活跃命令。当前唯一主线是
> T1ce -> WT={1,2,3}；执行边界见 `T1CE_MAINLINE.md`、
> `AFLOW_BRATS_HANDOFF_20260807.md` 与
> `AFLOW_BRATS_HANDOFF_20260810_ADDENDUM.md`。

首次日期：2026-06-22
最近更新：2026-07-25 04:20 CST（last4 partial-unfreeze 工程门通过，GPU smoke 等待控制面恢复）

本文档用于在新窗口继续 SAM-Med3D x BraTS2024 科研。**本轮起，科研主线发生方向性变更，
接续 agent 必须先读第 0 节。**

---

## 0. 给接续 agent 的最高优先级指令（必读，凌驾本文档其余所有内容）

### 2026-07-25 04:20 CST 当前最高优先级：工程门通过，先完成 GPU smoke，禁止绕过

matched rank-0 与 LoRA 的 clean-validation 结论不变。唯一允许的后续 arm 已冻结为从 untouched
base 新初始化、LoRA/target token 关闭、只解冻 image-encoder blocks 8--11，并继续训练
prompt+mask heads。两组 AdamW 分别为 `28,367,616` 参数、LR `1e-5` 与 `7,584,304` 参数、
LR `1e-4`，WD 均 `0.1`；总 trainable/model=`35,951,920/100,509,232`。最大 50 epochs，
epoch 20 前不早停，此后仅连续 20 个完整 epoch 没有 validation C3 strict new high 才停。

独立 helper/trainer/preflight/foreground launcher/checkpoint strict reload 已完成。最终审计补上了
matched arms 在模型与 optimizer 构建后的第二次 RNG reseed，并将
`rng_reseed_after_model_setup=true` 写入 machine prereg；修复后聚焦回归 `24/24 passed`，
更宽的 PEFT/T1ce-WT/partial-unfreeze CPU/synthetic 合约回归 `41/41 passed`，
CPU full preflight 通过且 `output_created=false`。human/machine prereg SHA-256 分别为
`111968f92c09f35272a5ddd35741c3ad1a5b2803a09f307b2a6993473e156156` /
`b55e763c4ab422a8b38d49df7fe6dde9e95af8d5e154d485eaef5fdcc97a5f91`，trainer SHA-256
`643c1e9124986ac4d057be3779d7eea5ff7b0e99009cea1361feb496f90cc20e`。

现在**不得直接启动 972/108 正式长训**。必须先在一张 BF16-capable GPU 上运行唯一
2-train/2-validation/e1 smoke：
`bash scripts/run_sammed3d_t1ce_wt_partial_unfreeze_smoke.sh`。本轮两次 workspace A10 请求与
一次 Arnold 请求都因 MLX control-plane HTTPS timeout 在分配前失败；没有 worker ID、
没有训练输出，smoke/formal 输出目录都仍不存在。控制面恢复后先 `worker list` 排除 orphan，
再只运行上述 smoke；须有限 loss/gradient/metric、记录显存/吞吐、完成 save + strict reload。
smoke 通过后才允许前台执行 formal launcher 的 `preflight` 与 `run`。

完整工程门与基础设施记录：
`workspace/BraTS/research_audits/FV_LAST4_PARTIAL_UNFREEZE_ENGINEERING_GATE_20260725.md`
（SHA-256 `afc25e55b81fa3d406a6037f261e54dd44a6596ddb8d65feda065d17c466d39a`）。
270-case test 继续永久冻结；不得访问、重跑或用于选择，也不得 threshold/TTA/morphology/
postprocess/full-encoder fine-tuning。

### 2026-07-25 03:37 CST 当前最高优先级：matched rank-0 已完成，只在 clean validation 推进 partial-unfreeze

`2026-07-24 10:36 CST` 要求的 matched whole-volume rank-0 decoder/prompt baseline 已按
972 train / 108 subject-clean validation、untouched base、whole-volume `128^3`、相同
train-only augmentation、3 次累计 corrective click、recurrent low-resolution mask 和 50 epochs
完整完成。rank-0 只训练 prompt encoder + mask decoder，image encoder 全冻结；selected
epoch 32 trainer validation C3=`0.492487`，checkpoint SHA-256
`7a25d0268799bdfffc626eacecb2a319a51d8127460143e7c6af3e8d4b532cf9`。

两臂随后在同一 CPU/runtime 上用独立 case-seeded PCG64 evaluator 严格 reload。rank-0 / LoRA
subject-balanced C3=`0.478505/0.521410`，LoRA-rank0=`+0.042905`；50-subject paired bootstrap
95% CI=`[+0.025863,+0.063080]`，42/50 subjects、81/108 scans 改善。scan-macro C3 delta=
`+0.042829`。这只支持“单 seed、validation-selected 的 conditional development matched
contrast”；同一 validation 同时选 epoch，且 worker 重启暴露非 bitwise determinism，禁止宣称
confirmatory、跨 seed、外部泛化或纯确定性 LoRA 因果效应。

完整结论与哈希边界：
`workspace/BraTS/research_audits/fv_lora_vs_rank0_matched_val108_conclusion_20260724.md`
（SHA-256 `3482801189e8055140aacadc70cd551ab673351e245b38c275eaa0235835fdbf`）。

下一步只允许在 972/108 clean validation 上预注册并实现独立命名的 image-encoder last-four-block
partial-unfreeze + discriminative-LR arm；只能解冻 blocks 8--11，严禁 full image-encoder
fine-tuning。必须先有 fail-closed 参数组/参数量/protocol/checkpoint metadata 与定向测试，再决定
是否启动。270-case test 继续永久冻结，不得访问、重跑或用于任何选择；仍禁止 threshold、TTA、
morphology、postprocess 和历史 inference-workflow 搜索。

### 2026-07-24 10:36 CST 当前最高优先级：冻结 test 结论，回到 clean validation 做受控改进

当前任务口径固定为 T1ce 单通道、binary official `WT={1,2,3}`（RC `{4}` 排除）、3 次累计
corrective click、recurrent low-resolution mask。FV-LoRA-Aug 正式 arm 已完整训练 50 epochs
并完成冻结的 270-case test；当前没有训练或评估进程，不得重复启动 watcher 或重跑 test。

正式 arm 从 untouched `sam_med3d_turbo.pth` 初始化，使用 whole-volume `128^3` resize、
train-only affine/elastic/Gaussian noise，以及 image encoder 12 blocks 的 Q/V LoRA-r4/a8；
encoder base weights 保持冻结，**只能写 encoder PEFT，禁止写 full image-encoder
fine-tuning**。972 train / 108 subject-clean validation、50 epochs，按 clean validation
选择 epoch 33=`0.532817`。冻结 270-case test 的 scan C1/C2/C3 =
`0.508103/0.516545/0.523030`，119-subject-balanced C3=`0.509143`。完整冻结、SHA 勘误、
报告和证据边界见最后第 67 节。

该 arm 没有达到 `0.7`，也低于旧 GT-crop v2b test C3 `0.553028`；但 geometry、初始化和
训练 recipe 同时变化，禁止将 `-0.029998` 归因于 LoRA、augmentation 或 whole-volume 中的
任一单因素。270-case test 已被多次项目内评估，不是 pristine external test；**从现在起不得用
其结果继续选模型、调学习率或比较新 arm，也不得再次运行。**

接续工作的正确顺序：

1. 只在 972 train / 108 clean validation 上建立 matched whole-volume 对照，至少补齐同一
   preprocessing/augmentation/training protocol 下的 rank-0 decoder/prompt baseline，避免拿
   GT-crop v2b 与 FV-LoRA 做伪单因素比较。
2. 使用独立 reload evaluator 核验 validation C1/C2/C3 和 subject-balanced C3；先形成 matched
   LoRA-vs-rank0 结论，再决定是否预注册 last-four-block partial-unfreeze + discriminative-LR arm。
3. 新训练只允许依据 clean validation 决策。不得访问 270-case test，不得做 threshold、TTA、
   morphology 或 postprocess 搜索。
4. MedTAWS/GRPO 是论文方法主线，但必须建立在冻结且有效的训练权重与同口径 validation
   baseline 上；当前不要回到历史 inference-workflow/threshold/TTA/postprocess 搜索。

当前 selected checkpoint：
`workspace/BraTS/sammed3d_t1ce_wt/fv_lora_r4_aug_train972_val108_e50_s20260725/sam_model_val_dice_best.pth`，
SHA-256
`c1d1bd79791aa7beeffc541095472dec968630cc70ea7967128c86552bd7f1ec`。
正式 test report SHA-256
`a087c357bbc71c60bba63cf179692266d2ebe19c64f58b44d8821986ae8ab13b`，
final binding SHA-256
`ae0f636f6246bb9ad4920b28bd3feefd568c56be4aa200812f131b3461c446f4`。

### 新 context 的最短阅读顺序

1. 先读本节最上方 `2026-07-24 10:36 CST` 当前最高优先级，取得当前任务、禁止事项和下一步。
2. 再读第 40/41 节，理解历史 GRPO 主线的正/负证据与不能夸大的边界。
3. 最后读第 61--67 节，理解 T1ce→WT 微调、v2b、whole-volume LoRA、冻结 test 和最新结论。
4. 需要论文口径时再查 `AFLOW_BRATS_RESULTS.md` 与 `AFLOW_BRATS_WEEKLY_REPORT.md` 的最后一节。

**本节下方所有日期为 2026-07-22 及更早、标题仍含“当前最高优先级”的小节均是按时间保留的
历史记录，没有当前执行权限；一律由最上方 2026-07-24 指令覆盖。**

### 2026-07-22 17:18 CST 当前最高优先级：撤销 0.011，正确三点击 DSC=0.330096

第 57 节 `0.011020` 丢失了 upstream recurrent low-res mask prompt，不是有效 SAM-Med3D
baseline，只能作 invalid-interface diagnostic。仓库 Phase-0 已在当前 B200 重跑，3 例历史
`tumor_WT` best-of-5 均值 `0.7194`，checkpoint/decoder/runtime 正常。

新 test 270 cases、T1ce、reconstructed residual `{1,2,3}`、upstream `ToCanonical +
GT-centered CropOrPad128` 下，用户规定的 uniform GT/FN/FP、累计 3 points，并同时传上一轮
low-res mask，C1/C2/C3 crop DSC=`0.243101/0.291748/0.330096`，best-of-3=`0.339958`。
完全 upstream RITM/current-point/recurrent-mask C3=`0.330285`。recurrent mask 在 matched
累计点 arm 上贡献 `+0.316002` C3 DSC，是原错误主因。

仍禁止将 `0.330096` 与 DPD-SAM `0.741` 宣称同协议：PDF 没有披露 residual raw-label mapping
或 automatic baseline prompt；仓库历史约 `0.716` 是 T2-FLAIR、WT `{1,2,3,4}`、5-click
best-of-5。报告 SHA-256 `18b6e245fe0a566ec7498d3f478b1d9e4405d4ae951b4b09e4031223ade37460`，
根因审计见 `workspace/BraTS/research_audits/sammed3d_official_interface_root_cause_20260722.md`。
详见第 58 节；单卡进程已退出。

**17:20 CST 命名核验**：`sammed3d/brats_metrics.py` 的 post-treatment official WT 与
`sammed3d/t1ce_binary.py` 的 residual target 都严格等于 raw `{1,2,3}`。因此上述 T1ce
270-case C3 `0.330096` 已经是 WT 结果，无需重复 GPU 推理。旧数据准备器中的历史
`tumor_WT={1,2,3,4}` 含 RC 且使用 T2-FLAIR，代码已明确标为 legacy-compatible，禁止混用。

### 2026-07-22 17:02 CST 当前最高优先级：用户冻结的三点击 baseline 已完成但未复现

> 已由上节纠错覆盖；本节无 recurrent mask 的数值不得再作 baseline。

用户规定 `num_clicks=3`：首点从 residual GT `{1,2,3}` 前景内随机采正点；后两点从上一轮
prediction XOR GT 中随机采，FN 正、FP 负；累积全部历史点再预测。相同 270-case test、T1ce、
原始未微调 `sam_med3d_turbo.pth` 已完成。Model-grid DSC click-1/2/3 为
`0.008049/0.010133/0.011020`，source-grid 为 `0.008079/0.010126/0.010966`。Click-3 对
click-1 平均 `+0.002970`，179 cases 改善、76 退化、15 相同。全部轨迹通过点击区域、标签与
累计点数合约核验。

该值远低于论文 prompted SAM-Med3D `0.741`，不得声称 baseline 复现。当前协议严格不传上一轮
mask prompt；upstream 3D validation 则使用 GT-centered 128-cubed ROI、当前点和 recurrent
low-res mask。下一步应做这些 interface/geometry 的单因素 sanity check，禁止 threshold/TTA/
postprocess 搜索。完整报告 SHA-256
`c495cf2626dbc7edcbeb17f2c9a72a8d16b68e2b2f633a73c4525c55637ad7c8`；评测器 SHA-256
`7e5c9a67a8ccb6a21e0bee5601b2d5e65d9ddd87936352a0e9f35f61737d1756`。仅用 B200 `1007515`，
进程已退出。详见第 57 节。

### 2026-07-22 16:52 CST 当前最高优先级：零训练自动 test baseline 已完成

用户已明确授权访问新 20% test。270 cases 已用原始未微调 `sam_med3d_turbo.pth`、仅 T1ce、整卷
`128^3` 完成 label-free baseline。固定 sigmoid 0.5，无 threshold/TTA/postprocess 搜索。严格
unprompted model-grid macro DSC 为 ET `0.002272`、residual `{1,2,3}` `0.006127`；仅由 T1ce
非零脑区 bbox 中心产生的自动正点为 `0.002231/0.005656`。ET present=209/270，residual
present=270/270。

这证明 promptable SAM-Med3D 在没有目标语义/目标定位提示时不能直接作为自动肿瘤分割器；该结果
不是论文 prompted SAM-Med3D `0.741` 的复现。下一步若做 sanity check，必须明确分成使用 GT
prompt 的 official/oracle 上界与完全 label-free automatic prompt baseline，禁止混为论文主结果。
完整逐例报告：`workspace/BraTS/t1ce_zero_shot_test/full_270_single_b200.json`，SHA-256
`88995830d20ba732f37fd2f1da04c7156aaacf7f164532431b07f608263f5273`；评测器 SHA-256
`609e9f1067c77c5002db7a42ee9b56d2c5802d33c8a50972bb38bf2dfdc9908a`。只使用 worker
`1007515`，现已退出且无 GPU compute。详见第 56 节。

### 2026-07-22 16:34 CST 当前最高优先级：只完成外层 80/20 切分

用户要求先停止模型工作，只对 1350 个 BraTS scans 做随机 80% train / 20% test。新冻结外层 split
按 subject 分组，固定 seed `20260722`，精确得到 train `1080 scans / 494 subjects`、test
`270 scans / 119 subjects`；case/subject overlap 均为 0，union=1350。test manifest 位于
`data/datasets/brats2024_random_subject_80_20_seed20260722_v1/sealed_test/`，构建与验证均未打开
test NIfTI。train/test manifest SHA-256 为
`aa2e55915e2a4bc7c2af2e95d6e08f2199397694fe4a92ea66999b534a22f63d` / 
`7b91690c63a5eb359be105b3cc7c4ad563d41fca2b2c84e521251b7ed006b8d2`。

同 seed 独立重建逐字节一致；6750 个原始 T1n/T1c/T2w/T2f/seg 路径全部存在。当前没有内部
validation split，未恢复 GPU 训练，ET-only 与 `{1,2,3}` residual-tumor target 仍待用户确认。
完整细节见第 55 节。

### 2026-07-22 16:21 CST 当前唯一执行口径

当前只做 **T1ce 单通道 -> `{1,2,3}` residual-tumor 二值 mask**。RC 标签 4 明确排除；四模态
P0/P0+、WT/TC/ET D3/D6 和旧 GT-click 结果均是历史，不得恢复。开发数据冻结为 872 fit / 209
validation、subject-disjoint；outer 269 继续封存。论文比较 headline 为 128-cubed model-grid
macro DSC/IoU/HD95，同时强制报告 `nearest-exact` 恢复后的 source-grid secondary；阈值固定 0.5，
禁止 threshold/TTA/postprocess 搜索。

全 1081 development 审计发现旧 categorical resize 使用 PyTorch legacy `nearest`，与 image
trilinear `align_corners=False` 的中心坐标约定不一致。GT source->128->source 自身 DSC mean 从
legacy `0.788933` 提升到 `nearest-exact 0.924792`。因此 16:00 前所有 T1ce smoke/pilot/checkpoint
已统一标 invalid、禁止 resume；失效清单位于
`workspace/BraTS/dpdsam_t1ce_b0/INVALID_LEGACY_NEAREST_ARTIFACTS_20260722.md`。v2 协议 SHA-256
`5feed4bb21f3c9d8f89e8fcd1d9ac48a74d71657409540d2048f69d386539cd4`。

corrected 256/64、20-epoch gate：B1 static prompt + encoder LoRA-r4 在 e20 得到 model/source DSC
`0.325470/0.325024`、model-grid IoU `0.209179`、HD95 `32.085 mm`；D1 frozen-encoder
K=4/rank-1 dynamic domain prompt 仅 `0.161892/0.161581`，不扩全量、不接 CDR。16:27 CST 用户
要求暂停并确认指标定义；全量 B1 停在 atomic epoch 1、尚无 validation，no-augmentation pilot
停在 epoch 8、e5 DSC `0.202683`。两张 B200 已核验无 compute，A2 runner 接入也已中止，未经
用户确认不得恢复。A2 是定位根因的工程臂，不冒充 DPD-SAM 论文实现。

完整测试为 `479 passed`。最近工程与审计见第 54 节。MedTAWS 只能在有效自动 baseline、权重、
dual-grid evaluator 和 checkpoint 先冻结后开始；当前不得启动搜索或访问 outer test。

### 2026-07-22 14:55 CST DPD-SAM 单模态纠正（历史，已由上节推进）

用户撤销四模态五类 Paper1913/P0 performance 主线，当前唯一任务定义改为 DPD-SAM 对齐：仅
T1ce（本地字段 `t1c`）单通道输入，一个 residual-tumor-vs-background 二值 mask，自动连续 prompt、
整体 volume resize 到 `128^3`，二值 DSC/IoU/HD95，patient-level 8:2 train/test 且 train 内约
20% val，固定同一 split 做 5 个独立 seed 并报告 mean+/-std。禁止继续把 WT/TC/ET D3 或六区域
D6 当主指标。

两条正在运行的四模态 P0/P0+ 已于 14:55 CST 收到 KeyboardInterrupt，B200 上均确认无残留
compute。P0 最后完整 atomic epoch 29、最后 full-184 validation epoch 20 D3 `0.647468`；P0+
最后完整 atomic epoch 17、最后 validation epoch 10 D3 `0.610302`。输出目录均新增
`TERMINATED_BY_DIRECTION_CHANGE.json` 且 `resume_authorized=false`，不得恢复。

本地主证据 `article/DPD-SAM.pdf` 明确论文 BraTS2024 SAM-Med3D baseline DSC
`74.1+/-1.5`、DPD-SAM `86.8+/-0.6`。新冻结草案：
`workspace/BraTS/training_protocols/dpdsam_t1ce_binary_restart_v0.md`。当前唯一 launch blocker 是
论文未公开说明 BraTS 五类标签折叠为 residual-tumor 二值 GT 的精确集合；不得自行猜 ET/TC/WT，
必须先从作者代码/证据恢复或由用户明确冻结。完整细节见最后第 53 节。

### 2026-07-22 Paper1913 对齐纠正（历史，已被上节撤销）

当前 performance 主线已经改为**四模态、全监督、自动分割**。MICCAI 2025 Paper1913 不是
SAM-Med3D prompt/click 实验；其 BraTS2024 3D U-Net/nnU-Net/Proposed D3 mean Dice 为
`0.8388/0.8588/0.8745`。A0/A1 的 `0.42-0.49` 是 GT-click binary region-query，只能作
interactive appendix，禁止再当 automatic baseline、禁止在其上继续 MedTAWS 搜索。

本轮已独立实现 P0 four-channel 3D U-Net、training-only tumor-centered `128^3` crop、无 GT
full-grid evaluator 与 conventional D3。500-step 两例 overfit D3 达 CE `0.889378` / Composite
`0.895191`；累计 764-step、8-case full-grid pilot 的 Composite D3 为 `0.513081`，显著优于 CE
`0.269754`。这证明原低分根因是错误任务接口与未训练权重，不是 BraTS 天生只能做到 0.4。

用户已建立长程 goal：完整 184-case tune conventional D3 首先达到 `0.80`，优先超过 Paper1913
3D U-Net `0.8388`，高分目标超过 nnU-Net `0.8588`；未达标必须继续逐因素训练诊断，不能用
pilot 代替正式结果。两张 B200 已从零并行启动 P0-Composite tumor-centroid 与独立命名的
P0+-Composite label-balanced(70% foreground / 30% brain-negative-center)，均为 764 fit、100
epochs、完整 184 tune 每 10 epochs。8-case pilot 不是正式结果；133 confirmatory、test、
additional labels 继续封存。自动 baseline/checkpoint/protocol 冻结前不得恢复 MedTAWS；恢复后仍
禁止 threshold/TTA/morphology/postprocess 搜索。完整细节见最后第 52 节。

### 2026-07-21 15:44 CST mask-token 语义审计纠正（历史基础，已被上节推进）

**立即停止第 47 节 frozen proposal-token baseline 的 tier2、confirmatory 和所有续跑。** 官方
SAM-Med3D 的四个 mask token 是 token0 single-mask 与 token1-3 ambiguity proposals，服务于同一
prompted binary target；它们不是 NETC/SNFH/ET/RC 四类。第 47 节把四 token 当类别、固定零背景
再做五类 argmax 的 contract 无效，`0.083186` 与搜索后的 `0.093116` 只保留为 invalid-interface
diagnostic，不能作为 SAM-Med3D baseline、MedTAWS 增益或论文结果。

三路独立审计确认 checkpoint strict load、四模态顺序/归一化/空间恢复、标签派生与 scorer 均无
额外压低 bug。相反，M1 的 `0.083186` 中 `0.067029` 来自 absent-GT+absent-pred 的 Dice=1，
占 80.6%；扣除该空区贡献后的 present-region 诊断约 `0.01616`。官方原生 checkpoint 复现是
单通道、逐二值目标、GT corrective-click、`multimask_output=False`/token0；本地旧复现为 WT
`0.716053`、SNFH `0.721915`、RC `0.455362`、TC `0.359991`、ET `0.333589`、NETC
`0.104110`，说明 checkpoint 没坏，但该 oracle-interactive 协议不能冒充自动多类结果。

代码已对旧 frozen eval、raw-token cache、tier1/tier2 MedTAWS 入口 fail-fast。新的最小自动底座
固定为四模态 target-prompt semantic adapter：每个原子类一个**可训练**类别提示，decoder 每类只走
官方 token0；推理不读 GT，但必须先在 fit split 监督训练。训练出有效 checkpoint 后，MedTAWS
才能搜索 multi-scale/coarse-to-refine/uncertainty routing。test/additional 271 和 133-case
confirmatory 继续封存。修正后的定向测试为 `35 passed`，完整 suite 为 `403 passed`。真实
checkpoint CPU strict-build 已确认 patch kernel `(768,4,16,16,16)`、target table `(4,384)`。
新 `m2` arm 已在 B200 完成两病例 train-save-strict-reload-full-grid smoke：loss `1.881118 ->
1.537241`，参数 delta norm `2.754253`，patch/decoder/target-prompt 梯度均有限非零，峰值显存
`6,651,082,240` bytes；两步后全背景，仅证明工程通路。报告 SHA-256
`940faba4a8464d9c5074dd2ab9227b8dc03feb8c5b1bbc19b473251c5b32f45c`。

### 2026-07-21 最新顺序纠正（覆盖本节及后文“先训练 checkpoint”的旧口径）

> **历史口径，已被上一小节撤销。** 不得再执行本小节的 frozen token baseline/search 顺序。

用户明确要求 **Work #1 先测、Work #2 后训练**。当前因果问题固定为：在同一份完全冻结、
不做任何梯度更新的 SAM-Med3D source checkpoint 上，四模态裸 baseline 与 MedTAWS 结构工作流
相比是否有 Dice 提升。`W/4` 四通道 inflation、C0 等权 gate 和 no-prompt decoder 接入是确定性
构造，不是训练。Work #1 主效应为：

`frozen M1 + MedTAWS - frozen M1 bare baseline`

先在 184-case / 84-subject `development_tune` 测 C0/M1 裸 baseline，再在同一 tune 上运行
MedTAWS / score-softmax / greedy / uniform 的同成本层、等真实 backend units 搜索。133-case /
67-subject confirmatory val 在最佳 workflow、search seeds、代码与统计规则冻结前禁止评分；test 与
additional 271 继续封存。Work #2 权重训练只能在 Work #1 的裸基线、搜索增益和失败分析落档后
另立协议启动。

Work #1 禁止 threshold、TTA、morphology、connected-component 或 postprocess 搜索。允许的全局
mask-token 到原子类排列必须在 tune 上选择后对所有 case 固定，并把 token mapping 收益与结构
路由收益分别报告；单病例执行时绝不能读取 GT。算法准确名称仍是 stochastic UCB-guided
workflow-tree search，不是标准 MCTS。

本轮协议：
`workspace/BraTS/training_protocols/brats2024_work1_frozen_fourmodal_medtaws_v1.md`。
该协议现已 REVOKED：`authorizes_weight_training=false`、`authorizes_development_tune_baseline=false`、
`authorizes_development_tune_search=false`、`authorizes_confirmatory_val=false`、
`authorizes_test_access=false`。

用户已明确要求不继续研究数据来源；当前开发实验按现有四模态数据相同处理。不要再把官方 zip
provenance 调研作为零训练 baseline、tune 搜索或后续开发训练的执行阻塞项；论文中仍保留既有
provenance 限制说明，不得伪称完成了未执行的 archive parity。

### 2026-07-20 四模态统一纠偏（本节最高优先级，覆盖后文旧口径）

用户已即时纠正方向：**从 work #1 开始，T1n、T1c、T2w、T2f 四个模态必须作为同一个
case 的联合输入。** 任何沿用“ET/TC/RC/NETC 只喂 T1c、WT/SNFH 只喂 T2f”的实验，均只
能作为历史记录，不能再进入当前主线或论文 headline。

新的三层依赖关系固定为：

1. **共同底座**：先建立四模态联合输入、全病例、自动四原子类输出的固定数据/模型/评测
   contract；输出 ET/NETC/SNFH/RC，再派生 TC 与不含 RC 的官方 WT。
2. **work #1 / MedTAWS**：算法不改，仍是基于 AFlow 的 UCB-guided workflow tree search，
   但要在冻结权重的四模态自动分割底座上重新验证。历史 single-modal、GT corrective-click、
   binary-target、threshold/TTA/postprocess rounds 只作 provenance，不作为新方法证据。新的
   workflow 搜索必须以结构/策略为对象，不得退回阈值、TTA、形态学网格。
3. **work #2 / 权重训练**：在与 work #1 完全相同的四模态输入、自动输出、官方 labels、
   subject-grouped folds 和 metrics 下训练/微调 SAM-Med3D，并与冻结权重底座及 nnU-Net/
   MedNeXt 强基线比较。

因此，“先 work #1”指先把 MedTAWS 的任务定义、输入输出、objective、operator、split、metric、
provenance 和 equal-budget ablation 全部审清并重建，不代表恢复旧的 inference threshold、TTA、
postprocess 搜索。当前 consumed sealed test 继续禁止读取或重试；additional 271 在最终配置冻结
前也不得访问标签。

### 当前唯一执行口径

1. **Work #1 不是“旧工程继续冻结复用”，而是“保留搜索算法原型，重建医学任务接入”。**
   MedTAWS 保留随机化 UCB-guided workflow tree search、父子轨迹和 AFlow provenance；旧
   `round_1` 至 `round_31` 的单模态/GT-click/binary backend 与全部 Dice 只读归档，不得作为
   新底座、推理外壳或论文主证据。
2. **先冻结共同 contract，再训练最小 semantic adapter。** source checkpoint 本身不存在可信的
   frozen/no-prompt/automatic 五类输出；必须先在 fit split 学习四个 target prompts 与 decoder，
   得到不读 GT 的有效 bare checkpoint。之后 Work #1 才能搜索 coarse-to-refine、multi-scale、
   uncertainty routing、跨模型/跨尺度融合等结构策略，不得再搜索 mask-token permutation。
3. **Work #2 与 Work #1 共用同一协议。** 输入固定为 `[T1n,T1c,T2w,T2f]`，原子标签固定为
   background/NETC/SNFH/ET/RC，`TC=NETC∪ET`，官方 `WT=NETC∪SNFH∪ET`。全病例进入评测，
   不删除 absent-class case；验证/推理不使用 label-centered crop，也不在推理时用 GT 生成 click。
   训练 loader 可在 train split 内使用 label-balanced patch sampling，但不得把这种采样 ROI 带入
   验证/推理或据此忽略完整网格假阳性。
4. **历史 v4 产物不能直接复用。** 它只保存 T1c/T2f、按二值 target 生成 positive-only rows、
   重采样到 1.5 mm，并默认 legacy WT `{1,2,3,4}`；新四模态主线必须建立独立 namespace，
   保留原始 1 mm 四模态共配准空间和原始五类 label。
5. **指标先于长训练冻结。** 主报告使用 BraTS 2024 lesion-wise Dice/HD95 的六区域等权汇总，
   subject-balanced/timepoint 汇总作为 secondary；任何搜索节点必须绑定 objective、fold、manifest、
   checkpoint、代码、operator、prompt、LLM/provider、RNG 和输出 SHA。

### 数据边界与历史结果

- 原始目录共有 1350 scans / 613 longitudinal subjects；subject id 是 `BraTS-GLI-xxxxx`，末尾
  `-100/-101/...` 是 timepoint。旧 200-scan split 有 subject leakage，所有旧结果只能追溯。
- 2026-07-17 建立的 948 train / 133 val / 269 sealed-test scan 分组可提供 subject provenance，
  但其单模态 binary manifest 不能进入新主线。该 sealed test 已被全局 ledger 消费，禁止重试、
  重命名或重新生成后复用。
- Synapse additional 271 的身份/标签版本尚未完成独立证明；在方法、checkpoint、metrics 和统计
  全部冻结前禁止访问其标签，也不得预先称为官方 hidden test。
- 第 40 节 `label_RC=0.58932` 是旧 scan-level、单模态、GT-click、subject-leaking 结果；第 43 节
  的 subject-clean partial-decoder 结果仍是单模态、GT-click、binary 结果。二者只能进入
  interactive appendix，不能与官方自动分割 leaderboard 比较。
- 当前 2 x NVIDIA B200 可用。完成数据 provenance、四模态 contract、no-GT evaluator 和最小
  CPU/GPU smoke 后，优先并行跑相同 seed 的强 baseline 与 SAM 配对臂，再补独立 seed。

### 立即执行顺序

1. 用已通过 B200 smoke 的 `m2` target-prompt contract 建立 fit/tune 正式训练协议。
2. 在 764-case fit 训练；184-case tune 只做 checkpoint/训练配置选择，先报告逐区与空区分解。
3. 冻结一个不读 GT 的有效 bare checkpoint 后，才运行 MedTAWS 与 greedy/softmax/uniform 等预算
   结构搜索；禁止 mask-token permutation、threshold、TTA、morphology 与 postprocess 搜索。
4. 冻结最佳 workflow 与代码/统计后，才在 133-case confirmatory val 做一次配对确认。
5. 仅在最终方法、checkpoint 和统计全部冻结后，讨论一次性外部 holdout。

2026-07-21 进度覆盖：公开 scorer parity、C0/M1 两病例训练通路 smoke 和 paired fit/tune split
已完成，详见第 46 节。第 47 节 frozen token baseline 已撤销；`m2` B200 真实 smoke 已通过，当前
无 GPU compute process。下一项是冻结正式 fit/tune 训练协议并训练有效 semantic checkpoint。

> 本文第 1 至 43 节均为历史实验账本；凡与本节冲突，一律以本节和第 44 节后的新记录为准。

---

## 1. 本阶段训练任务定义

- **模型**：SAM-Med3D（promptable 3D 分割模型，冻结权重的 zero-shot 阶段已结束，现在要训练它）。
  保留 click-prompt 协议；训练时用迭代 click 模拟（见仓库内 `get_next_click3D_torch_ritm`
  那条 RITM 点击采样链路）。
- **微调策略：parameter-efficient 优先，full fine-tune 后置。** 理由：3D 全量微调显存/时间/
  稳定性成本高，先用便宜方案确认信号，再升级。升级阶梯见第 3 节 milestone。
  - 阶梯 A（最便宜）：冻结 image encoder，只训 mask decoder（+ prompt encoder）。
  - 阶梯 B（中等）：在 image encoder 注意力层加 LoRA / adapter，让表征适配 BraTS。
  - 阶梯 C（最贵）：full fine-tune image encoder，小学习率 + 强 augmentation。
- **目标优先级**：
  - 主攻**弱目标**：`tumor_TC`、`tumor_ET`、`label_RC`、`label_NETC`（zero-shot 只有
    0.10~0.45，这是真正的 headroom，也是训练最能体现价值的地方）。
  - **约束目标**：`tumor_WT`、`label_SNFH`（zero-shot 已接近天花板 ~0.72，训练后**不能退化**，
    作为不退化约束，不是优化对象）。
- **损失**：Dice + CE/focal（BraTS 标准组合，建议直接用 MONAI `DiceCELoss` 一类实现）。
  主指标仍用 Dice，保持与历史结果可比；论文需要可再加 HD95 等，但不替换 Dice 作为主轴。
- **prompting 协议**：训练和评估必须用**同一套** click/prompt 策略（点数、采样方式固定），
  否则训练前后不可比。早期就把它定死，不要中途改。

注意（来自历史记录）：`get_next_click3D_torch_ritm` 当前 **batch 不安全**（只用第一个
batch 元素算 distance transform）。若训练要 `batch > 1`，必须先把 click sampler 改成
batch-safe，否则梯度信号是错的。先用 `batch=1` 或修好 sampler 再加大 batch。

---

## 2. 数据与划分（科研严谨性硬要求，最容易翻车的地方）

**致命前提**：work #1 阶段所有后处理增益，都是在 full eval cases（以及各 target 的
full 子集）上**搜索**出来的。这意味着这些 case 已经被 work #1 "看过/调过"。但当前本地
preprocessed 数据的现实是：`brats_full_tumor_WT.jsonl` 等 full JSONL 基本覆盖了各 target
目录下的全部病例（例如 WT 200/200，RC 168/168，NETC 67/67）。因此，在**不引入额外
BraTS 数据**的前提下，"test 与所有历史 eval / 后处理 grid search case 严格不相交"这个要求
事实上不可实现。

当前本地 preprocessed 数据规模：

| Target | 当前本地病例数 | 对应 full JSONL 行数 | 备注 |
|---|---:|---:|---|
| `tumor_WT` | 200 | 200 | full JSONL 覆盖全部本地 WT |
| `label_SNFH` | 200 | 200 | full JSONL 覆盖全部本地 SNFH |
| `label_RC` | 168 | 168 | full JSONL 覆盖全部本地 RC |
| `tumor_TC` | 146 | 146 | full JSONL 覆盖全部本地 TC |
| `tumor_ET` / `label_ET` | 145 | 145 | full JSONL 覆盖全部本地 ET |
| `label_NETC` | 67 | 67 | 病例数最少，训练/验证划分要特别小心 |

所以本阶段的严谨做法要分两种情况：

1. **如果能拿到额外 BraTS cases**：优先建立真正 external holdout。这个 holdout 不参与 work #1、
   不参与训练、不参与 val，只在最终验收时使用。这是最强论文口径。
2. **如果只能使用当前本地 preprocessed 数据**：不要声称 test 与 work #1 历史 eval 完全不相交。
   改为建立新的 **patient-level train / val / test split** 或 K-fold cross-validation。这个 split
   对 work #2 训练阶段是干净的：train 用于训练，val 用于选 checkpoint / 早停 / 调超参，test fold
   不参与训练阶段任何决策。
3. split 必须按纵向 subject 前缀 `patient_id=BraTS-GLI-xxxxx` 做全局划分，**不能按完整
   `case_id=BraTS-GLI-xxxxx-yyy` 分组**，也不能按 target 文件各自随机划分。完整 case id
   的末尾是 timepoint；同一 subject 的任何 timepoint 或任何 target 跨 split 都属于病人级泄漏。
4. 每个 target 的 train/val/test JSONL 都从同一个全局 case split 派生：该 target 有 label 的病例才写入
   对应 target JSONL。
5. 所有"训练 vs zero-shot"的对比，必须在**同一个 test fold**上、用**同一套 eval harness**
   （`eval-file` / `eval-multi-target`）、配**同一个固定 workflow 推理外壳**，做 apples-to-apples。

论文写法上要诚实区分：

- work #1：inference-time workflow optimization，历史 full eval 已被用于后处理搜索。
- work #2：training/fine-tuning，在新建 patient-level split 或 K-fold 上比较
  zero-shot + fixed workflow vs trained checkpoint + fixed workflow。

如果做不到 external holdout，至少做 K-fold，并在报告里写清楚划分方式。论文级结论必须经得起
"训练集和测试集是否真的分开"这个最基本的拷问。

---

## 3. 执行节奏：防漂移 milestone（每步都要有可验证产物）

不要一上来铺开几十个训练配置，也不要直接冲 full fine-tune。按 milestone 推进；M0-M2 只使用
subject-clean train/val，**test 一直封存到最终配置冻结后的 M3**。

- **M0 — 训练通路 sanity（最先做）**
  目标：训练 loop 能在一个极小子集（例如 4~8 个 case）上**过拟合**——loss 持续下降到接近 0、
  小集 Dice 飙到很高。这证明数据 pipeline / 梯度 / loss / click 协议全部打通。
  产物：可跑的训练脚本 + 过拟合曲线/日志。**这一步不通过，不许往下走。**

- **M1 — PEFT 微调（阶梯 A，decoder/prompt-encoder only）**
  在干净 train 上做最便宜的 PEFT，主攻弱目标。
  验收：弱目标（先挑一个最弱且最有代表性的，例如 `label_RC` 或 `tumor_TC`）在 clean val 上的 Dice
  **明显**超过 zero-shot + 固定 workflow；同时 WT/SNFH **不退化**。
  产物：checkpoint + 训练配置 + clean-val 对比表。

- **M2 — 升级（阶梯 B/C）**
  只有 M1 出现明确正信号，才升级到 image encoder LoRA / full fine-tune，或扩展到多个弱目标
  联合训练。每次升级都要回到同一个 clean val 证明确实更好，而不是更复杂。

- **M3 — 固化为 thesis work**
  保存最终 checkpoint + 完整训练配置 + test 集对比表（训练 vs zero-shot，逐 target Dice），
  作为毕业论文中"训练/微调"这条工作的结果。补 holdout / 交叉验证让结论站住。

纪律说明：**这里的"先聚焦单个弱目标再扩展"和 work #1 的"WT 单目标"不是一回事。** work #1 的
单目标纪律已随 work #1 一起冻结。本阶段的聚焦对象是**弱目标的训练**，不是 WT 的后处理。

---

## 4. 让 Codex 自主实现的部分（用户已授权自行完成）

- **先盘点仓库已有训练代码再动手**：SAM-Med3D 体系通常自带训练脚本/迭代 click 训练逻辑。
  先在 `sammed3d/`（含 `engine.py`，work #1 已在此加过推理方法）和仓库其余位置查找已有的
  train loop / trainer / loss / 数据加载，能复用就复用，缺什么再写，不要从零重造。
- **训练入口对齐现有 wrapper 习惯**：建议在 `scripts/run_worker_research.sh` 里新增/确认一个
  `train` 子命令，使训练像现有 `eval-file` / `grid-postprocess` 一样可在 worker 上一行启动。
  期望接口形如（具体参数由你定义并在脚本里实现）：

  ```bash
  bash scripts/run_worker_research.sh train \
    <train_jsonl> <val_jsonl> <output_dir> \
    --strategy decoder_only|lora|full \
    --targets label_RC,tumor_TC \
    --epochs ... --lr ... --batch-size 1
  ```

- **评估入口必须支持加载训练 checkpoint**：现有 workflow 默认会加载原始
  `sam_med3d_turbo.pth`。训练阶段必须新增一种干净方式，让 `eval-file` / `eval-multi-target`
  在不改后处理参数的情况下加载训练出来的 checkpoint，例如：
  - 给 `sammed3d.engine.get_model()` 增加 checkpoint override / model cache key；
  - 或用环境变量指定 checkpoint 路径；
  - 或新建固定 workflow round，仅改变 checkpoint 来源，不改变 workflow 后处理外壳。
  没有 checkpoint-aware eval，训练结果就无法和 zero-shot 做 apples-to-apples 对比。

- 自主决定：PEFT 具体注入位置、lr schedule、augmentation、checkpoint/早停策略、混合精度、
  梯度检查点（3D 显存吃紧时）等。
- 唯一要求：每个 milestone 都留下**可验证产物**（脚本 / 日志 / checkpoint / test 对比表），
  并且评估走干净 test 集 + 现有 eval harness。

---

## 5. 明确禁止项（防止 Codex 退回老路）

1. 不再把 `grid-postprocess` 后处理参数搜索当作主任务（仅可在训练已固化后，作为已冻结外壳调用）。
2. 不把前处理/后处理调参的小幅 Dice 波动当作本阶段成果汇报。
3. **不在训练用过的数据（train/val）上报告最终分数**；最终分数只看干净 test 集。
4. 不一次性铺开大量训练配置；按 M0→M1→M2→M3 升级。
5. 不在没过 M0 sanity 的情况下直接烧 H100 跑长训练。

---

## 6. work #1 已完成记录（冻结，仅作参考与推理外壳）

work #1（inference-time / no-training workflow optimization）已完成，下列结论冻结，**不再继续优化**，
仅供：(a) 论文写 work #1 时引用；(b) 训练后作为固定推理外壳复用。

- full-target 已完整报告最佳：`round_18`。
- WT 单目标新候选：`round_31`，full `tumor_WT` = `0.73482`（较 round_18 `+0.00672`）。
- RC mirror TTA：`round_30`，full `label_RC` = `0.44957`（较 round_18 `+0.01972`）。
- TC/NETC：`round_23` 的 full-validated 分支。

`round_18` full per-target（作为 zero-shot + workflow 的基线，训练要打败的就是它）：

| Target | Cases | Baseline | round_18 |
|---|---:|---:|---:|
| `tumor_WT` | 200 | 0.71605 | 0.72810 |
| `tumor_TC` | 146 | 0.35999 | 0.35661 |
| `tumor_ET` | 145 | 0.33359 | 0.33552 |
| `label_SNFH` | 200 | 0.72191 | 0.72304 |
| `label_RC` | 168 | 0.45536 | 0.42985 |
| `label_NETC` | 67 | 0.10411 | 0.10106 |

固定推理外壳文件（训练后复用，不要再改其参数）：

```text
workspace/BraTS/workflows/round_31/graph.py   # WT 分支 + RC mirror TTA + TC/NETC 继承
workspace/BraTS/workflows/round_30/graph.py   # RC mirror TTA
workspace/BraTS/workflows/round_23/graph.py   # TC/NETC full-validated 分支
workspace/BraTS/workflows/round_18/...         # full-target reported baseline
```

LLM/AFlow search（work #1 的自动搜索层）状态：rounds 24/26/27/28/29 均未超过 round_23，
该自动搜索方向已收尾，本阶段不继续。详见 `AFLOW_BRATS_RESULTS.md`。

更多 work #1 细节见：

```text
AFLOW_BRATS_RESULTS.md
AFLOW_BRATS_WEEKLY_REPORT.md
../BASELINE_RESULTS.md
```

---

## 7. 仍然有用的环境与命令参考

worker / 评估 harness 在训练阶段仍要用。

当前 H100 worker（2026-06-29 `mlx worker list` 查到）：`975543`。旧 worker `967020` 已过期。
如要重启长任务，先在 `975543` 上确认环境（probe），不要沿用旧 worker 状态。

H100 探针结果：

```text
GPU: NVIDIA H100 80GB HBM3
CUDA available: True
```

config：`config/config2.yaml` 是 git-ignored 本地文件，含 AIDP crawl key；不要在日志/文档/
提交信息中回显任何 key。

探针：

```bash
mlx worker login 975543 -- bash /mlx_devbox/users/songjunyi.666/playground/AFlow/scripts/run_worker_research.sh probe
```

在干净 test 集上做多目标评估（训练 vs zero-shot 的统一验收口径）：

```bash
mlx worker login 975543 -- bash /mlx_devbox/users/songjunyi.666/playground/AFlow/scripts/run_worker_research.sh \
  eval-multi-target <workflow_round> \
  /mlx_devbox/users/songjunyi.666/playground/AFlow/workspace/BraTS/full_eval/<run_name> \
  tumor_TC,tumor_ET,label_RC,label_NETC
```

单文件评估：

```bash
mlx worker login 975543 -- bash /mlx_devbox/users/songjunyi.666/playground/AFlow/scripts/run_worker_research.sh \
  eval-file <workflow_round> \
  /mlx_devbox/users/songjunyi.666/playground/AFlow/data/datasets/<clean_test>.jsonl \
  /mlx_devbox/users/songjunyi.666/playground/AFlow/workspace/BraTS/<out_dir>
```

> 后处理 grid 命令（`grid-postprocess` 系列）属于 work #1，已冻结。本阶段不要把它当主任务跑；
> 如确需调用固定外壳，仅用已固化配置，不做参数搜索。

---

## 8. 本阶段下一步（接续 agent 直接从这里开始）

1. **读第 0 节，确认本阶段做训练，不做后处理调参。**
2. 盘点仓库已有训练代码（`sammed3d/`、`scripts/`），决定复用还是新建 train loop。
3. 在 BraTS2024 当前数据上建立 patient-level train/val/test 划分或 K-fold。若没有额外病例，
   不要声称 test 与 work #1 历史 eval 完全不相交；要保证它对 work #2 训练阶段不泄漏，
   并固定 seed、落盘 JSONL。
4. 实现 checkpoint-aware eval：训练 checkpoint 必须能通过现有 eval harness 加载，并且只替换
   模型权重，不改变固定 workflow 后处理外壳。
5. 完成 **M0 sanity**：训练 loop 在极小子集上过拟合，证明通路正确。
6. 跑 **M1 PEFT**（decoder-only 起步），主攻一个最弱代表目标（如 `label_RC` 或 `tumor_TC`），
   在干净 test 上对比 zero-shot + 固定 workflow，并确认 WT/SNFH 不退化。
7. 有正信号再进 **M2** 升级（LoRA / full / 多弱目标联合），最终 **M3** 固化 checkpoint + 配置 +
   test 对比表，作为论文中"训练"这条工作的结果。

---

## 9. 2026-06-29 GRPO-style segmentation 方向进展

用户新增方向：希望把 GRPO 这类强化学习训练范式迁移到 SAM-Med3D/BraTS
分割权重训练上。注意：这不是恢复 work #1 后处理搜索；GRPO 方向必须真的更新模型权重。

已新增脚本：

```text
scripts/train_sammed3d_grpo.py
scripts/run_worker_research.sh grpo-train ...
```

当前实现口径：

- 一个 BraTS case + target 作为一个 query。
- 对同一病例采样 K 个 segmentation rollouts。
- rollout 随机性来自不同 click seed / prompt trajectory。
- reward 由 Dice、空 mask 惩罚、体积异常惩罚、碎片化惩罚组成。
- 对同一病例内部做 group-relative advantage，不训练 critic。
- 用 Bernoulli mask log-prob 的 PPO/GRPO-style surrogate 更新
  `prompt_encoder + mask_decoder`。
- 加 reference KL penalty 和少量 supervised anchor，避免 RL 初期发散。
- 默认仍冻结 image encoder。

已跑 smoke / audit：

| Run | 目的 | Target | Cases | Group size | 结果 |
|---|---|---|---:|---:|---|
| `workspace/BraTS/grpo/audit_label_RC_2case/` | 离线 reward audit | `label_RC` | 2 val | 4 | mean reward 0.25698，reward std 0.12995 |
| `workspace/BraTS/grpo/smoke_label_RC_2case/` | 1 epoch GRPO-style 训练 smoke | `label_RC` | 2 train / 2 val | 2 | 训练、反传、checkpoint 保存跑通 |
| `workspace/BraTS/grpo/audit_label_RC_2case_g2_zeroshot/` | 同 seed zero-shot audit | `label_RC` | 2 val | 2 | reward 0.24888，Dice 0.24802 |
| `workspace/BraTS/grpo/audit_label_RC_2case_g2_trained/` | 同 seed trained audit | `label_RC` | 2 val | 2 | reward 0.43717，Dice 0.41253 |
| `workspace/BraTS/grpo/audit_label_RC_4val_m0init_g2/` | M0 supervised checkpoint audit | `label_RC` | 4 val | 2 | reward 0.41310，Dice 0.35743 |
| `workspace/BraTS/grpo/label_RC_m0init_8train_4val_e3/` | M0 初始化 GRPO-style 3 epoch 短训练 | `label_RC` | 8 train / 4 val | 2 | best val reward 0.47624，best val Dice 0.41064 |
| `workspace/BraTS/grpo/audit_label_RC_4val_grpo_e3_g2/` | 同 seed GRPO checkpoint audit | `label_RC` | 4 val | 2 | reward 0.47504，Dice 0.40753 |

解释：这些仍是小样本工程 smoke，不能作为论文结论。它证明 GRPO-style pipeline
不是空转：reward group 有方差，训练能反传保存 checkpoint，同 seed audit 下训练后
checkpoint 有正向信号。下一步要扩大到 clean val，并和同规模 supervised
decoder-only 对照，而不是直接报 test 结论。

推荐下一步：

1. 在 `label_RC` 上跑 8-16 train cases / 4-8 val cases 的 GRPO-style short run，
   仍使用 fixed patient-level split。
2. 同步跑一个 supervised decoder-only 同规模对照，区分 GRPO-style 的增益是否超过
   DiceCE supervised anchor。
3. 只有 clean val 有稳定正信号后，才在 test fold 上做 zero-shot vs supervised PEFT
   vs GRPO-style PEFT 的固定 workflow 对比。

---

## 10. 2026-06-30 系统盘清理后状态

本次清理目标是释放 128G 系统盘空间，同时保留继续训练/评估所需的核心资产。

清理前后：

```text
/dev/vdn  125G  116G  9.5G  93%
/dev/vdn  125G  100G   26G  80%
```

已删除内容：

- `workspace/BraTS/cache` 下的推理轨迹缓存。
- 失败或纯 smoke 产物：`m0_label_RC_overfit1`、`smoke_label_RC`、
  `grpo/smoke_label_RC_2case`。
- 重复 checkpoint：只保留有效 run 的 `sam_model_val_dice_best.pth`，
  删除同 run 下的 `latest` / `val_loss_best` / `val_reward_best` 副本。
- Python `__pycache__`、并行评估 shard 临时目录、code-server 旧 VSIX 缓存、
  Claude/Merlin 旧版本扩展。

明确保留内容：

- `brats2024` 原始数据，约 42G。
- `sammed3d-venv` 环境。
- `SAM-Med3D-main` 的预处理数据与基础 checkpoint。
- 当前有效训练产物：
  - `workspace/BraTS/training/m0_label_RC_overfit4/sam_model_val_dice_best.pth`
  - `workspace/BraTS/grpo/label_RC_m0init_8train_4val_e3/sam_model_val_dice_best.pth`

当前科研可以继续。建议下一步不要动 test fold，先扩大 clean val 上的
GRPO-style short run，并补一个同规模 supervised decoder-only 对照：

```bash
bash scripts/run_worker_research.sh grpo-train \
  data/datasets/brats_patient_split_seed2026/brats_train_label_RC.jsonl \
  data/datasets/brats_patient_split_seed2026/brats_val_label_RC.jsonl \
  workspace/BraTS/grpo/label_RC_m0init_16train_8val_e5 \
  --targets label_RC \
  --max-train-cases 16 \
  --max-val-cases 8 \
  --group-size 2 \
  --epochs 5 \
  --num-clicks 5 \
  --num-workers 0 \
  --lr 0.00002 \
  --supervised-anchor 0.10 \
  --kl-coef 0.02 \
  --checkpoint workspace/BraTS/training/m0_label_RC_overfit4/sam_model_val_dice_best.pth \
  --reference-checkpoint workspace/BraTS/training/m0_label_RC_overfit4/sam_model_val_dice_best.pth
```

完成 clean val 对比后，再进入固定 test fold 的 zero-shot vs supervised PEFT
vs GRPO-style PEFT 公平评估。

---

## 11. 2026-06-30 GRPO clean 16/8 short run

本节记录 work #2 训练主线的最新结果，仍然不涉及后处理搜索。

注意：`workspace/BraTS/grpo/label_RC_m0init_16train_8val_e5/` 这个同名目录被前一次
中断 run 后续写入污染，`history.csv` 有两段 5-epoch 记录。不要用它作为正式结论。
后续使用下面这个干净目录：

```text
workspace/BraTS/grpo/label_RC_m0init_16train_8val_e5_clean_20260630/
```

配置：`label_RC`，M0 supervised checkpoint 初始化，16 train cases / 8 val cases，
5 epoch，group size 2，5 clicks，`lr=2e-5`，`supervised_anchor=0.10`，`kl_coef=0.02`。

同一 8-val、同 seed、同 GRPO audit-only 口径结果：

| Checkpoint | Run | Reward | Dice |
|---|---|---:|---:|
| M0 supervised init | `audit_label_RC_8val_m0init_g2` | 0.28598 | 0.26830 |
| Supervised 16/8 e5 best | `audit_label_RC_8val_sup16_e5_dicebest_g2` | 0.29478 | 0.28344 |
| GRPO clean 16/8 e5 best | `audit_label_RC_8val_grpo16_clean_e5_best_g2` | 0.31907 | 0.29956 |

训练脚本内部 clean-val 最优：

| Run | Best epoch | Val reward | Val Dice |
|---|---:|---:|---:|
| `label_RC_m0init_16train_8val_e5_clean_20260630` | 5 | 0.31436 | 0.29749 |
| `label_RC_m0init_16train_8val_e5_sup` | 2 for Dice | - | 0.29627 |

解释：GRPO-style PEFT 在这个小规模 clean-val 设置下有正信号，且在统一 audit-only
口径下超过同规模 supervised 对照。但样本仍小，不能作为 test 或论文结论。下一步应该
先做 32 train / 16 val 或多 seed 复核；只有 clean val 仍稳定优于 supervised，才进入
固定 test fold 的 zero-shot vs supervised PEFT vs GRPO-style PEFT 公平评估。

---

## 12. 2026-06-30 GRPO 32/16 clean-val 复核

目标：在比 16/8 更稳一点的 clean-val 设置上复核 GRPO-style PEFT 是否优于同规模
supervised decoder-only PEFT。仍只用 patient-level train/val split，不碰 test fold。

配置：`label_RC`，M0 supervised checkpoint 初始化，32 train cases / 16 val cases，
5 epoch，5 clicks，seed 2026。

训练内最优：

| Run | Best epoch | Val reward | Val Dice |
|---|---:|---:|---:|
| `grpo/label_RC_m0init_32train_16val_e5_clean_20260630` | 3 | 0.47461 | 0.42189 |
| `training/label_RC_m0init_32train_16val_e5_sup` | 3 | - | 0.43343 |

同一 16-val、同 seed、同 GRPO audit-only 口径：

| Checkpoint | Run | Reward | Dice |
|---|---|---:|---:|
| M0 supervised init | `audit_label_RC_16val_m0init_g2` | 0.30776 | 0.29950 |
| Supervised 32/16 e5 best | `audit_label_RC_16val_sup32_e5_best_g2` | 0.47535 | 0.42186 |
| GRPO 32/16 e5 best | `audit_label_RC_16val_grpo32_e5_best_g2` | 0.47818 | 0.42782 |

解释：32/16 下 GRPO 和 supervised 都显著超过 M0。GRPO 在统一 audit-only 口径下
仍略高于 supervised，但优势很小（reward +0.00282，Dice +0.00596），不能直接写成
强结论。下一步优先做多 seed 复核，确认这个小优势是否稳定；稳定后再考虑 test fold。

---

## 13. 2026-06-30 GRPO 32/16 多 seed 复核

目的：验证 32/16 设置下 GRPO-style PEFT 相对 supervised PEFT 的小优势是否稳定。
样本子集不变，seed 只影响 dataloader shuffle、click/rollout 随机性与 audit rollout。

同一 16-val、同 seed、同 GRPO audit-only 口径：

| Seed | Checkpoint | Reward | Dice |
|---:|---|---:|---:|
| 2026 | M0 init | 0.30776 | 0.29950 |
| 2026 | Supervised 32/16 best | 0.47535 | 0.42186 |
| 2026 | GRPO 32/16 best | 0.47818 | 0.42782 |
| 2027 | M0 init | 0.30773 | 0.29950 |
| 2027 | Supervised 32/16 best | 0.47651 | 0.42676 |
| 2027 | GRPO 32/16 best | 0.46706 | 0.41756 |

两 seed 均值：

| Method | Reward mean | Dice mean |
|---|---:|---:|
| M0 init | 0.30775 | 0.29950 |
| Supervised 32/16 best | 0.47593 | 0.42431 |
| GRPO 32/16 best | 0.47262 | 0.42269 |

结论更新：GRPO-style pipeline 已经证明能真实更新权重，并且稳定超过 M0 起点；但在
32/16 多 seed clean-val 复核下，GRPO 尚未稳定超过同规模 supervised decoder-only。
当前不能写成“GRPO 优于 supervised”。更准确的说法是：GRPO 训练范式已迁移成功，
但需要改进 reward/rollout/group 配置，才能争取超过强 supervised baseline。

建议下一步：优先做 GRPO 方法学 ablation，而不是上 test fold：

1. 增大 group size 到 4，降低 group-relative advantage 噪声。
2. 调低 supervised anchor（如 0.03 或 0.05），确认 RL reward 是否提供独立增益。
3. 固定 32/16 clean-val，不动 test fold；只有多 seed 稳定超过 supervised 后再 test。

---

## 14. 2026-06-30 当前实验结论摘要

当前能成立的结论：

1. 训练主线已经从 inference-time workflow / 后处理搜索切换到权重训练，且 patient-level
   split、checkpoint-aware eval、M0 supervised sanity、GRPO-style trainer 都已跑通。
2. GRPO-style segmentation trainer 不是空转：它会真实更新 `prompt_encoder + mask_decoder`，
   并在 `label_RC` clean-val 上稳定超过 M0 起点。
3. 在 32 train / 16 val / seed 2026 和 2027 的复核中，GRPO-style PEFT 与 supervised
   decoder-only PEFT 都显著超过 M0。
4. 当前不能声称 GRPO 优于 supervised。两 seed 均值下 supervised Dice 0.42431，GRPO Dice
   0.42269，GRPO 没有稳定超过强 supervised baseline。

当前不能做的事：

- 不能上 test fold 选择 checkpoint 或调 reward。
- 不能继续做阈值、component-size、TTA、grid-postprocess 等 work #1 后处理搜索。
- 不能在论文里写“GRPO 显著优于 supervised”；只能写“GRPO 训练范式已迁移成功，但仍需方法学改进”。

当前下一步科研问题：如何让 GRPO 的 group-relative reward 优化提供超过 supervised
DiceCE 的独立增益。优先 ablation：增大 group size、降低 supervised anchor、重审 reward penalty。

---

## 15. 2026-06-30 GRPO g4 anchor0.05 ablation seed 2026

目的：验证增大 group size、降低 supervised anchor 是否能让 GRPO 提供超过 supervised
baseline 的独立增益。仍固定 `label_RC`、32 train / 16 val，不碰 test fold。

配置：M0 checkpoint 初始化，`group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`，
5 epoch，seed 2026。

训练内最优：

| Run | Best epoch | Val reward | Val Dice |
|---|---:|---:|---:|
| `grpo/label_RC_m0init_32train_16val_e5_g4_anchor005_seed2026` | 4 | 0.48755 | 0.43557 |

同配置 reload audit-only：

| Checkpoint | Reward | Dice |
|---|---:|---:|
| GRPO g4 anchor0.05 seed2026 best | 0.50245 | 0.44541 |

对比既有 seed2026 16-val audit：

| Method | Reward | Dice |
|---|---:|---:|
| M0 init | 0.30776 | 0.29950 |
| Supervised 32/16 best | 0.47535 | 0.42186 |
| GRPO g2 anchor0.10 best | 0.47818 | 0.42782 |
| GRPO g4 anchor0.05 best | 0.50245 | 0.44541 |

解释：在 seed 2026 上，`group_size=4 + supervised_anchor=0.05` 明显优于之前的
GRPO g2 和 supervised baseline，是当前最强 clean-val 结果。但这仍只是单 seed，
下一步必须跑 seed 2027 同配置验证稳定性。

---

## 16. 2026-06-30 GRPO g4 anchor0.05 多 seed 复核

目的：复核 `group_size=4 + supervised_anchor=0.05` 是否稳定优于 supervised baseline。
仍固定 `label_RC`、32 train / 16 val，不碰 test fold。

同一 16-val、同 seed、同 group size 4 audit-only 口径：

| Seed | Method | Reward | Dice |
|---:|---|---:|---:|
| 2026 | Supervised 32/16 best | 0.47535 | 0.42186 |
| 2026 | GRPO g2 anchor0.10 best | 0.47818 | 0.42782 |
| 2026 | GRPO g4 anchor0.05 best | 0.50245 | 0.44541 |
| 2027 | Supervised 32/16 best | 0.47651 | 0.42676 |
| 2027 | GRPO g2 anchor0.10 best | 0.46706 | 0.41756 |
| 2027 | GRPO g4 anchor0.05 best | 0.48876 | 0.43855 |

两 seed 均值：

| Method | Reward mean | Dice mean |
|---|---:|---:|
| Supervised 32/16 best | 0.47593 | 0.42431 |
| GRPO g2 anchor0.10 best | 0.47262 | 0.42269 |
| GRPO g4 anchor0.05 best | 0.49561 | 0.44198 |

结论更新：`group_size=4 + supervised_anchor=0.05` 是当前第一个在两个 seed 上都超过
同规模 supervised baseline 的 GRPO 配置。它相对 supervised 的两 seed 平均增益为
reward +0.01968、Dice +0.01767。仍然不能上 test fold；下一步要做单变量 ablation，
拆分 group size 与 supervised anchor 的贡献。

---

## 17. 2026-06-30 GRPO anchor 单变量 ablation

目的：保持 `group_size=2` 不变，只把 `supervised_anchor` 从 0.10 降到 0.05，判断低 anchor
是否本身带来增益。设置仍为 `label_RC`、32 train / 16 val、seed 2026。

训练内最优：

| Run | Best epoch | Val reward | Val Dice |
|---|---:|---:|---:|
| `grpo/label_RC_m0init_32train_16val_e5_g2_anchor005_seed2026` | 4 | 0.47303 | 0.42287 |

reload audit-only：

| Method | Reward | Dice |
|---|---:|---:|
| GRPO g2 anchor0.10 seed2026 | 0.47818 | 0.42782 |
| GRPO g2 anchor0.05 seed2026 | 0.49447 | 0.43290 |
| GRPO g4 anchor0.05 seed2026 | 0.50245 | 0.44541 |

结论：降低 supervised anchor 在 `group_size=2` 下也有正向贡献，但仍不如
`group_size=4 + anchor0.05`。下一步需要跑 `group_size=4 + anchor0.10`，拆分 group size
本身的贡献。

---

## 18. 2026-06-30 GRPO group-size 单变量 ablation

目的：保持 `supervised_anchor=0.10` 不变，只把 `group_size` 从 2 提到 4，判断 group size
本身的贡献。设置仍为 `label_RC`、32 train / 16 val、seed 2026。

训练内最优：

| Run | Best epoch | Val reward | Val Dice |
|---|---:|---:|---:|
| `grpo/label_RC_m0init_32train_16val_e5_g4_anchor010_seed2026` | 4 | 0.49070 | 0.43592 |

reload audit-only：

| Method | Reward | Dice |
|---|---:|---:|
| GRPO g2 anchor0.10 seed2026 | 0.47818 | 0.42782 |
| GRPO g2 anchor0.05 seed2026 | 0.49447 | 0.43290 |
| GRPO g4 anchor0.10 seed2026 | 0.49462 | 0.43922 |
| GRPO g4 anchor0.05 seed2026 | 0.50245 | 0.44541 |

结论：group size 4 与低 supervised anchor 都有贡献。低 anchor 在 g2 下主要提升 reward，
group size 4 对 Dice 提升更明显；两者叠加得到当前最佳 seed2026 结果。下一步如果继续，
优先复核 `g4 anchor0.10` 的 seed2027，或尝试 `g4 anchor0.03`，但系统盘余量已经偏低，
建议先清理重复 checkpoint 再继续大跑。

---

## 19. 2026-06-30 GRPO ablation 后系统盘清理

目的：为继续做 GRPO 训练 ablation 腾出空间，同时保留所有可复现实验结论需要的最优权重。

清理结果：

| Item | Before | After |
|---|---:|---:|
| `/` used | 114G | 104G |
| `/` avail | 12G | 22G |
| `/` use% | 92% | 84% |

已删除：

- 受污染旧目录：`workspace/BraTS/grpo/label_RC_m0init_16train_8val_e5/`。
- GRPO/training run 中重复或中间态 checkpoint：`sam_model_latest.pth`、`sam_model_val_reward_best.pth`、`sam_model_val_loss_best.pth`。
- `workspace/BraTS` 下的 `__pycache__`。

已保留：

- 所有有效 run 的 `sam_model_val_dice_best.pth`。
- M0 checkpoint、supervised baselines、GRPO g2/g4 ablation 的 best checkpoint。
- 原始数据、SAM-Med3D base checkpoint、venv 均未触碰。

当前可以继续科研。下一步优先补 `group_size=4 + supervised_anchor=0.10` 的 seed 2027，
用来判断 group size 单变量贡献是否跨 seed 稳定；仍只在 clean-val 上比较，不碰 test fold。

---

## 20. 2026-06-30 GRPO group-size 单变量多 seed 复核

目的：补齐 `group_size=4 + supervised_anchor=0.10` 的 seed 2027，判断只增大 group size
是否能跨 seed 稳定超过 supervised baseline。设置仍为 `label_RC`、32 train / 16 val、M0 初始化。

seed 2027 训练内最优：

| Run | Best epoch | Val reward | Val Dice |
|---|---:|---:|---:|
| `grpo/label_RC_m0init_32train_16val_e5_g4_anchor010_seed2027` | 3 | 0.48080 | 0.43155 |

seed 2027 reload audit-only：

| Method | Reward | Dice |
|---|---:|---:|
| Supervised 32/16 seed2027 | 0.47651 | 0.42676 |
| GRPO g2 anchor0.10 seed2027 | 0.46706 | 0.41756 |
| GRPO g4 anchor0.10 seed2027 | 0.48917 | 0.43618 |
| GRPO g4 anchor0.05 seed2027 | 0.48876 | 0.43855 |

两 seed reload audit-only 均值：

| Method | Reward mean | Dice mean |
|---|---:|---:|
| Supervised 32/16 best | 0.47593 | 0.42431 |
| GRPO g2 anchor0.10 best | 0.47262 | 0.42269 |
| GRPO g4 anchor0.10 best | 0.49190 | 0.43770 |
| GRPO g4 anchor0.05 best | 0.49561 | 0.44198 |

结论更新：`group_size=4` 本身已经是稳定正贡献；在 `anchor=0.10` 下两 seed 均超过
supervised baseline，Dice 平均增益约 +0.01339。`anchor=0.05` 在 group size 4 上继续带来
小幅增益，Dice 均值比 `anchor=0.10` 高约 +0.00428，因此当前优先配置仍是
`group_size=4 + supervised_anchor=0.05`。下一步可以只在 seed 2026 上探索更低 anchor
（如 0.03）是否过度偏离 supervised anchor；若不提升，就收敛到 g4 anchor0.05 做更大样本。

---

## 21. 2026-07-03 GRPO anchor0.03 小范围探索与接手状态

目的：在当前最佳区域附近做一次低 anchor 探索，判断 `supervised_anchor=0.03` 是否比
`0.05` 更好。设置仍固定为 `label_RC`、32 train / 16 val、M0 初始化、`group_size=4`、
seed 2026；仍只用 clean-val，不碰 test fold。

训练内最优：

| Run | Best epoch | Val reward | Val Dice |
|---|---:|---:|---:|
| `grpo/label_RC_m0init_32train_16val_e5_g4_anchor003_seed2026` | 4 | 0.49120 | 0.43719 |

reload audit-only：

| Method | Reward | Dice |
|---|---:|---:|
| GRPO g4 anchor0.10 seed2026 | 0.49462 | 0.43922 |
| GRPO g4 anchor0.05 seed2026 | 0.50245 | 0.44541 |
| GRPO g4 anchor0.03 seed2026 | 0.49803 | 0.44097 |

结论：`anchor=0.03` 没有超过 `anchor=0.05`，不建议继续扩到 seed 2027。
当前主配置仍是 `group_size=4 + supervised_anchor=0.05`。

交接状态：

- 当前没有继续推理后处理搜索；work #1 的 threshold/component-size/TTA/grid-postprocess 线保持冻结。
- 当前有效科研主线是 work #2：训练/微调 SAM-Med3D 权重，并固定同一 eval harness、公平对比。
- 当前 clean-val 最强配置：`GRPO g4 anchor0.05`，两 seed Dice mean 0.44198；supervised 32/16 两 seed Dice mean 0.42431。
- `g4 anchor0.10` 两 seed Dice mean 0.43770，说明 group size 4 本身稳定有效。
- `anchor0.03` 是负结果，已记录，不继续扩 seed。
- test fold 尚未用于这些 ablation；下一窗口不要直接上 test，除非先明确冻结配置。
- 系统盘清理过；重复 `latest/reward/loss` checkpoint 已删，保留各 run 的 `sam_model_val_dice_best.pth`。

建议下一步：把 `g4 anchor0.05` 从 32 train 扩到更大 train cases（例如 64 或 96），仍用同一
patient-level split、同一 clean-val、同一 reload audit-only 口径；若更大样本稳定，再考虑固定配置后上 held-out test fold。

---

## 22. 2026-07-03 GRPO g4 anchor0.05 扩样本到 64/96 train

目的：按第 21 节交接建议，把当前主配置 `group_size=4 + supervised_anchor=0.05`
从 32 train cases 扩到更大训练样本。仍固定 `label_RC`、M0 初始化、16 val、5 epoch、
同一 patient-level split、同一 clean-val、同一 reload audit-only 口径；test fold 仍未使用。

多卡执行状态：

- 训练脚本没有单进程多卡 / DDP / DataParallel 支持；不要强行改训练范式。
- 本轮用 2 张 A10 并行跑两个独立进程：seed 2026 用 `cuda:0`，seed 2027 用 `cuda:1`。
- 直接用 `CUDA_VISIBLE_DEVICES=...` 在 sandbox 内启动会触发 PyTorch CUDA 初始化失败并回退 CPU；
  正确做法是走已批准的 `bash scripts/run_worker_research.sh grpo-train ... --device cuda:0/1` 入口。

64 train / 16 val 训练内最优：

| Seed | Run | Best epoch | Val reward | Val Dice |
|---:|---|---:|---:|---:|
| 2026 | `grpo/label_RC_m0init_64train_16val_e5_g4_anchor005_seed2026` | 5 | 0.56586 | 0.49641 |
| 2027 | `grpo/label_RC_m0init_64train_16val_e5_g4_anchor005_seed2027` | 4 | 0.60033 | 0.52152 |

64 train reload audit-only：

| Seed | Audit dir | Reward | Dice |
|---:|---|---:|---:|
| 2026 | `grpo/audit_label_RC_16val_grpo64_g4_anchor005_best_g4_seed2026` | 0.58815 | 0.50979 |
| 2027 | `grpo/audit_label_RC_16val_grpo64_g4_anchor005_best_g4_seed2027` | 0.59218 | 0.51516 |
| mean | - | 0.59016 | 0.51247 |

96 train / 16 val 训练内最优：

| Seed | Run | Best epoch | Val reward | Val Dice |
|---:|---|---:|---:|---:|
| 2026 | `grpo/label_RC_m0init_96train_16val_e5_g4_anchor005_seed2026` | 5 | 0.61287 | 0.53394 |
| 2027 | `grpo/label_RC_m0init_96train_16val_e5_g4_anchor005_seed2027` | 5 | 0.62954 | 0.54855 |

96 train reload audit-only：

| Seed | Audit dir | Reward | Dice |
|---:|---|---:|---:|
| 2026 | `grpo/audit_label_RC_16val_grpo96_g4_anchor005_best_g4_seed2026` | 0.61406 | 0.53206 |
| 2027 | `grpo/audit_label_RC_16val_grpo96_g4_anchor005_best_g4_seed2027` | 0.62120 | 0.54348 |
| mean | - | 0.61763 | 0.53777 |

和 32-case best / supervised baseline 的公平 clean-val 对比：

| Method | Train / Val | Reward mean | Dice mean |
|---|---:|---:|---:|
| Supervised 32/16 best | 32 / 16 | 0.47593 | 0.42431 |
| GRPO g4 anchor0.05 best | 32 / 16 | 0.49561 | 0.44198 |
| GRPO g4 anchor0.05 best | 64 / 16 | 0.59016 | 0.51247 |
| GRPO g4 anchor0.05 best | 96 / 16 | 0.61763 | 0.53777 |

结论更新：

- 64 train 已稳定优于 32 train：Dice mean 0.51247，相对 32-case g4 anchor0.05 的 0.44198
  提升 +0.07049。
- 96 train 进一步优于 64 train：Dice mean 0.53777，相对 64-case 提升 +0.02530，
  相对 supervised 32/16 的 0.42431 提升 +0.11346。
- 当前 clean-val 最强配置变为 `label_RC`、M0 初始化、`group_size=4`、
  `supervised_anchor=0.05`、96 train / 16 val、5 epoch。
- 当前仍不能把结论写成 test 结果；test fold 尚未用于这些 GRPO ablation。

清理状态：

- 本轮 64/96 四个训练目录都只保留 `sam_model_val_dice_best.pth`。
- 已删除本轮重复的 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`。
- 两张 A10 已空闲；清理后 `/` 约 109G used / 17G avail / 87%。

建议下一步：

1. 若继续扩样本，优先同配置跑 full train（当前 split 下 `label_RC` train 共 118 cases），仍用同一 16-val audit-only 口径。
2. 若要进入 test fold，先明确冻结配置（例如冻结为 96 train / 16 val 或 full train 后的最终 checkpoint），再只做一次 held-out test 对比；不要在 test 上继续选 epoch、调 reward 或调后处理。

---

## 23. 2026-07-03 GRPO g4 anchor0.05 full train 与文献启发下一步

目的：延续第 22 节的扩样本正信号，把同一主配置扩到当前 split 下的全部
`label_RC` train cases（118 train / 16 val）。配置仍固定为 M0 初始化、
`group_size=4`、`supervised_anchor=0.05`、5 epoch、同一 patient-level split、
同一 clean-val、同一 reload audit-only 口径；test fold 仍未使用。

full train / 16 val 训练内最优：

| Seed | Run | Best epoch | Val reward | Val Dice |
|---:|---|---:|---:|---:|
| 2026 | `grpo/label_RC_m0init_fulltrain_16val_e5_g4_anchor005_seed2026` | 5 | 0.64104 | 0.55336 |
| 2027 | `grpo/label_RC_m0init_fulltrain_16val_e5_g4_anchor005_seed2027` | 5 | 0.62052 | 0.54254 |

full train reload audit-only：

| Seed | Audit dir | Reward | Dice |
|---:|---|---:|---:|
| 2026 | `grpo/audit_label_RC_16val_grpo118_g4_anchor005_best_g4_seed2026` | 0.63555 | 0.55016 |
| 2027 | `grpo/audit_label_RC_16val_grpo118_g4_anchor005_best_g4_seed2027` | 0.62623 | 0.54380 |
| mean | - | 0.63089 | 0.54698 |

和已有 clean-val 结果的公平对比：

| Method | Train / Val | Reward mean | Dice mean |
|---|---:|---:|---:|
| Supervised 32/16 best | 32 / 16 | 0.47593 | 0.42431 |
| GRPO g4 anchor0.05 best | 32 / 16 | 0.49561 | 0.44198 |
| GRPO g4 anchor0.05 best | 64 / 16 | 0.59016 | 0.51247 |
| GRPO g4 anchor0.05 best | 96 / 16 | 0.61763 | 0.53777 |
| GRPO g4 anchor0.05 best | 118 / 16 | 0.63089 | 0.54698 |

结论更新：

- 当前 clean-val 最强配置更新为 `label_RC`、M0 初始化、`group_size=4`、
  `supervised_anchor=0.05`、118 train / 16 val、5 epoch。
- full train Dice mean 0.54698，相对 96 train 的 0.53777 提升 +0.00921，
  相对 32-case g4 anchor0.05 的 0.44198 提升 +0.10500，
  相对 supervised 32/16 的 0.42431 提升 +0.12267。
- 两个 full-train seed 的 best epoch 都在 epoch 5，说明扩样本后尚未出现明显过拟合；
  若继续只在 clean-val 内推进，可考虑小幅延长到 7-8 epoch，但这不是优先级最高的结构性 idea。
- 当前结果仍不能写成 test 结果；test fold 尚未用于这些 GRPO ablation。

清理状态：

- `label_RC_m0init_fulltrain_16val_e5_g4_anchor005_seed2026` 和
  `label_RC_m0init_fulltrain_16val_e5_g4_anchor005_seed2027` 都只保留
  `sam_model_val_dice_best.pth`。
- 已删除两个 full-train 目录中的 `sam_model_latest.pth` 和
  `sam_model_val_reward_best.pth`。
- 两张 A10 当前空闲。

文献启发的下一步（仍只走 clean-val，不碰 test）：

1. **组内偏好优化 / DPO-style ablation**：SAMPO、SAM+DPO 和 MAPO 都说明
   SAM 类医学分割可以从偏好优化中受益。最贴近当前代码的做法是保留
   `group_size=4 + anchor0.05`，在每个 rollout group 内用 reward 最高和最低样本构造
   best-vs-worst pair，增加 pairwise preference loss；先跑 32 或 64 train smoke，
   再扩 full train。
2. **dropout/stochastic hypothesis 多样性**：MAPO 强调用随机假设产生偏好对。
   当前 group diversity 主要来自 click/randomness；下一步可以只在训练 rollout 中打开
   decoder dropout 或轻量 mask perturbation，eval/reload audit 不改。
3. **空间奖励小项**：MedGround-R1 和 staged voxel-level RL 都把语义/空间一致性放进 reward。
   当前 reward 以 Dice 为主，可做小系数 boundary/surface Dice 或 center-of-mass reward ablation；
   这比调 postprocess 更符合 work #2，因为优化对象仍是权重。
4. **置信度/边界 anchor**：probabilistic-aware weak supervision 的思路可以缝合到
   supervised anchor 上，对边界不确定区域降权，减少 RC 边界噪声对 anchor 的牵制。

参考方向：

- SAM-Med3D: https://arxiv.org/abs/2310.15161
- SAMPO: https://arxiv.org/abs/2508.02464
- Enhancing SAM with Efficient Prompting and Preference Optimization: https://arxiv.org/abs/2503.04639
- MAPO: https://arxiv.org/abs/2512.15009
- MedGround-R1: https://arxiv.org/abs/2507.02994
- Staged Voxel-Level Deep Reinforcement Learning: https://arxiv.org/abs/2601.03875
- Interactive 3D medical segmentation with multi-agent RL: https://arxiv.org/abs/1911.10334
- Probabilistic-aware weakly supervised 3D segmentation: https://arxiv.org/abs/2403.02566

---

## 24. 2026-07-03 DPO-style pairwise preference ablation 负结果

目的：根据第 23 节的 SAMPO / SAM+DPO / MAPO 启发，尝试把 group 内 reward 排序转成
best-vs-worst preference signal。仍固定 `label_RC`、M0 初始化、32 train / 16 val、
`group_size=4`、`supervised_anchor=0.05`、同一 clean-val、同一 reload audit-only 口径；
test fold 未使用。

代码改动：

- `scripts/train_sammed3d_grpo.py` 新增显式参数
  `--objective {grpo,pairwise,grpo_pairwise}`，默认仍是 `grpo`，旧实验路径不变。
- `pairwise`：用组内 reward 最高/最低 rollout 构造 DPO-style
  `-log sigmoid(beta * ((logp_best - logp_worst) - ref_margin))`。
- `grpo_pairwise`：保留原 GRPO policy loss，只额外加
  `preference_coef * pairwise_preference_loss`；本轮用 `preference_coef=0.05`、
  `preference_beta=8.0`、`preference_pairs=best_worst`。
- 8 train / 4 val / 1 epoch smoke 已跑通，只作为链路检查，不作为科研结论。

纯 pairwise objective，32 train / 16 val，训练内 best：

| Seed | Best epoch | Val reward | Val Dice |
|---:|---:|---:|---:|
| 2026 | 3 | 0.46923 | 0.41846 |
| 2027 | 2 | 0.47245 | 0.42469 |

纯 pairwise reload audit-only：

| Seed | Audit dir | Reward | Dice |
|---:|---|---:|---:|
| 2026 | `grpo/audit_label_RC_16val_pairwise32_g4_anchor005_beta8_best_seed2026` | 0.46155 | 0.41357 |
| 2027 | `grpo/audit_label_RC_16val_pairwise32_g4_anchor005_beta8_best_seed2027` | 0.47395 | 0.42373 |
| mean | - | 0.46775 | 0.41865 |

GRPO + pairwise auxiliary，32 train / 16 val，训练内 best：

| Seed | Best epoch | Val reward | Val Dice |
|---:|---:|---:|---:|
| 2026 | 4 | 0.48349 | 0.42810 |
| 2027 | 3 | 0.49196 | 0.43928 |

GRPO + pairwise auxiliary reload audit-only：

| Seed | Audit dir | Reward | Dice |
|---:|---|---:|---:|
| 2026 | `grpo/audit_label_RC_16val_grpo_pairwise32_g4_anchor005_coef005_beta8_best_seed2026` | 0.47542 | 0.42429 |
| 2027 | `grpo/audit_label_RC_16val_grpo_pairwise32_g4_anchor005_coef005_beta8_best_seed2027` | 0.48063 | 0.43048 |
| mean | - | 0.47803 | 0.42739 |

公平对比：

| Method | Train / Val | Reward mean | Dice mean |
|---|---:|---:|---:|
| Supervised 32/16 best | 32 / 16 | 0.47593 | 0.42431 |
| GRPO g4 anchor0.05 best | 32 / 16 | 0.49561 | 0.44198 |
| Pure pairwise beta8 | 32 / 16 | 0.46775 | 0.41865 |
| GRPO + pairwise coef0.05 beta8 | 32 / 16 | 0.47803 | 0.42739 |
| GRPO g4 anchor0.05 best | 118 / 16 | 0.63089 | 0.54698 |

结论：

- 纯 pairwise 低于 supervised 和旧 GRPO，不扩样本。
- 小系数 hybrid 略高于 supervised，但仍明显低于旧 GRPO 32-case；seed2027 一度接近旧 GRPO，
  但 seed2026 不稳定，均值没有竞争力，不扩样本。
- 当前 best-vs-worst pairwise 写法可能过度惩罚低 reward mask 的整幅 threshold action；
  对医学分割这种 dense voxel action，rank-only pairwise 没有保留 advantage magnitude，
  还可能放大 group 内坏样本噪声。
- 当前 clean-val 主结论不变：最强仍是 `GRPO g4 anchor0.05, 118 train / 16 val`
  reload audit Dice mean 0.54698。

清理状态：

- pure pairwise、hybrid、smoke 训练目录都只保留 `sam_model_val_dice_best.pth`。
- 已删除本轮重复的 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`。
- 两张 A10 已空闲。

建议下一步：

1. 不继续扩 pure pairwise 或 `grpo_pairwise coef0.05`。
2. 若继续缝合论文 idea，优先做更贴近 segmentation geometry 的训练侧 reward：
   boundary/surface Dice 或 center-of-mass 小系数辅助 reward；不要回到 test-time postprocess。
3. 也可以先在当前 best full-train 配置上做 7-8 epoch clean-val 复核，因为 full-train 两 seed 的
   best 都在 epoch 5；但这属于训练时长 ablation，不应优先于几何 reward idea。

---

## 25. 2026-07-03 center-of-mass spatial reward ablation 负结果

目的：继续尝试第 23 节的空间奖励方向，但选择比 surface distance 更低成本的
center-of-mass alignment。该项仍只影响训练/eval reward，不改 inference workflow、
threshold、postprocess 或 test fold。设置固定为 `label_RC`、M0 初始化、32 train / 16 val、
`group_size=4`、`supervised_anchor=0.05`、5 epoch、同一 clean-val、同一 reload audit-only。

代码改动：

- `scripts/train_sammed3d_grpo.py` 新增 `--center-reward-coef`，默认 `0.0`，旧 GRPO 默认行为不变。
- 当 `center_reward_coef > 0` 时，reward 额外加
  `center_reward_coef * center_alignment(pred, gt)`。
- 本轮使用 `--center-reward-coef 0.05`。注意 reward 数字包含新增中心项，
  不宜和旧 reward 均值直接硬比；主要比较 Dice。

center reward 32 train / 16 val 训练内 best：

| Seed | Best epoch | Val reward | Val Dice |
|---:|---:|---:|---:|
| 2026 | 4 | 0.53352 | 0.42983 |
| 2027 | 3 | 0.52725 | 0.42790 |

center reward reload audit-only：

| Seed | Audit dir | Reward | Dice |
|---:|---|---:|---:|
| 2026 | `grpo/audit_label_RC_16val_center32_g4_anchor005_center005_best_seed2026` | 0.52918 | 0.42642 |
| 2027 | `grpo/audit_label_RC_16val_center32_g4_anchor005_center005_best_seed2027` | 0.53212 | 0.43188 |
| mean | - | 0.53065 | 0.42915 |

Dice 对比：

| Method | Train / Val | Dice mean |
|---|---:|---:|
| Supervised 32/16 best | 32 / 16 | 0.42431 |
| GRPO g4 anchor0.05 best | 32 / 16 | 0.44198 |
| Pure pairwise beta8 | 32 / 16 | 0.41865 |
| GRPO + pairwise coef0.05 beta8 | 32 / 16 | 0.42739 |
| GRPO + center reward coef0.05 | 32 / 16 | 0.42915 |
| GRPO g4 anchor0.05 best | 118 / 16 | 0.54698 |

结论：

- center reward 比 pure pairwise / hybrid pairwise 略好，且略高于 supervised；
  但仍低于原始 GRPO g4 anchor0.05 的 32-case Dice mean 0.44198，不扩样本。
- center-of-mass 奖励太粗，只能鼓励位置靠近，不能直接优化边界或局部 overlap；
  在当前 RC 任务上没有超过简单 Dice reward + volume/component penalty 的组合。
- 当前 clean-val best 不变：`GRPO g4 anchor0.05, 118 train / 16 val`
  reload audit Dice mean 0.54698。

清理状态：

- center reward 两个训练目录都只保留 `sam_model_val_dice_best.pth`。
- 已删除本轮重复的 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`。
- test fold 仍未用于这些 ablation。

建议下一步：

1. 不扩 center reward coef0.05。
2. 若继续训练侧 idea，优先考虑 full-train best 的 7-8 epoch clean-val 延长复核；
   这是当前最直接、风险最低的增量，因为 full-train 两 seed best 都在 epoch 5。
3. 几何 reward 若继续做，应避免 coarse center reward，可考虑更直接的 boundary/surface Dice，
   但要先做计算成本 smoke；仍不要回到 inference/postprocess。

---

## 26. 2026-07-06 原始 GRPO full-train 延长到 8 epoch，成为当前 clean-val best

目的：复核第 23 节 full-train 5 epoch 是否欠训练。仍固定 work #2 训练权重主线，
不改 inference workflow / threshold / component-size / TTA / postprocess，不使用 test fold。

设置：

- target：`label_RC`
- 初始化：M0 checkpoint `workspace/BraTS/training/m0_label_RC_overfit4/sam_model_val_dice_best.pth`
- objective：原始 `grpo`
- train / val：118 / 16，同一 patient-level split，同一 clean-val
- `group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`，`num_clicks=5`
- epoch：从 full-train 5 epoch 延长到 8 epoch

8 epoch 训练内 best：

| Seed | Run | Best epoch | Val reward | Val Dice |
|---:|---|---:|---:|---:|
| 2026 | `grpo/label_RC_m0init_fulltrain_16val_e8_g4_anchor005_seed2026` | 7 | 0.67535 | 0.58189 |
| 2027 | `grpo/label_RC_m0init_fulltrain_16val_e8_g4_anchor005_seed2027` | 8 | 0.65702 | 0.57021 |
| mean | - | - | 0.66618 | 0.57605 |

8 epoch reload audit-only：

| Seed | Audit dir | Reward | Dice |
|---:|---|---:|---:|
| 2026 | `grpo/audit_label_RC_16val_grpo118_e8_g4_anchor005_best_seed2026` | 0.66248 | 0.57084 |
| 2027 | `grpo/audit_label_RC_16val_grpo118_e8_g4_anchor005_best_seed2027` | 0.65290 | 0.56763 |
| mean | - | 0.65769 | 0.56923 |

和已有 clean-val reload audit-only 的公平对比：

| Method | Train / Val | Epochs | Reward mean | Dice mean |
|---|---:|---:|---:|---:|
| Supervised best | 32 / 16 | 5 | 0.47593 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 5 | 0.49561 | 0.44198 |
| GRPO g4 anchor0.05 | 64 / 16 | 5 | 0.59016 | 0.51247 |
| GRPO g4 anchor0.05 | 96 / 16 | 5 | 0.61763 | 0.53777 |
| GRPO g4 anchor0.05 | 118 / 16 | 5 | 0.63089 | 0.54698 |
| GRPO g4 anchor0.05 | 118 / 16 | 8 | 0.65769 | 0.56923 |

结论：

- 按当时的旧 scan-level split，clean-val best 更新为 `label_RC`、M0 初始化、原始 GRPO、
  `group_size=4`、`supervised_anchor=0.05`、118 train / 16 val、8 epoch。
- 8 epoch reload Dice mean 0.56923，相对 5 epoch full-train 的 0.54698 提升 +0.02225；
  相对 96 train 的 0.53777 提升 +0.03146；相对 32-case GRPO 的 0.44198 提升 +0.12725；
  相对 supervised 32/16 的 0.42431 提升 +0.14492。
- 两 seed 均提升，且 seed2027 best 在 epoch 8，说明 5 epoch full-train 明显偏短；
  下一步优先做原始 GRPO 的更长训练/训练效率优化，而不是继续扩 pairwise 或 center reward。
- 当前结果仍是 clean-val 结论；test fold 仍未用于这些 GRPO ablation。若上 test，
  必须先冻结配置。

清理状态：

- 两个 8 epoch 训练目录均只保留 `sam_model_val_dice_best.pth`。
- 已删除 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`。

下一步建议：

1. 在 A100 80GB 上优先提高单进程训练吞吐和显存利用：先审计 batch/click sampler 安全性；
   若 batch>1 仍不安全，则用更长 epoch 或更高 `group_size` 的单卡实验吃显存。
2. 原始 GRPO 主线优先跑 12 epoch 或 10-12 epoch 早停复核，仍只用 clean-val/reload audit。
3. 若要做新 idea，先小成本做 boundary/surface reward，而不是继续 pure pairwise、
   grpo_pairwise coef0.05 或 center-of-mass reward。

---

## 27. 2026-07-06 A100 后续推进：原始 GRPO full-train 10 epoch 成为当前 clean-val best

背景：机器切到单张 A100-SXM4-80GB。已开两个只读子 agent 审计训练系统与科研下一步。
结论是当前 GRPO 脚本为单进程单卡，不支持 DDP；`DataLoader` 硬编码 `batch_size=1`，
且 `batch_value()` / reward logging 按单样本假设写，因此 batch>1 不安全。A100 上优先用
`num_workers=2` 提高吞吐，科研变量仍保持原始 GRPO 主线，不碰 test fold。

设置：

- target：`label_RC`
- 初始化：M0 checkpoint `workspace/BraTS/training/m0_label_RC_overfit4/sam_model_val_dice_best.pth`
- objective：原始 `grpo`
- train / val：118 / 16，同一 patient-level split，同一 clean-val
- `group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`，`num_clicks=5`
- epoch：10
- 工程参数：`num_workers=2`，单 A100，单进程单卡

10 epoch 训练内 best：

| Seed | Run | Best epoch | Val reward | Val Dice |
|---:|---|---:|---:|---:|
| 2026 | `grpo/label_RC_m0init_fulltrain_16val_e10_g4_anchor005_nw2_seed2026` | 10 | 0.69022 | 0.59192 |
| 2027 | `grpo/label_RC_m0init_fulltrain_16val_e10_g4_anchor005_nw2_seed2027` | 9 | 0.68534 | 0.58999 |
| mean | - | - | 0.68778 | 0.59095 |

10 epoch reload audit-only：

| Seed | Audit dir | Reward | Dice |
|---:|---|---:|---:|
| 2026 | `grpo/audit_label_RC_16val_grpo118_e10_g4_anchor005_nw2_best_seed2026` | 0.69118 | 0.59415 |
| 2027 | `grpo/audit_label_RC_16val_grpo118_e10_g4_anchor005_nw2_best_seed2027` | 0.67832 | 0.58450 |
| mean | - | 0.68475 | 0.58932 |

和已有 clean-val reload audit-only 的公平对比：

| Method | Train / Val | Epochs | Reward mean | Dice mean |
|---|---:|---:|---:|---:|
| Supervised best | 32 / 16 | 5 | 0.47593 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 5 | 0.49561 | 0.44198 |
| GRPO g4 anchor0.05 | 64 / 16 | 5 | 0.59016 | 0.51247 |
| GRPO g4 anchor0.05 | 96 / 16 | 5 | 0.61763 | 0.53777 |
| GRPO g4 anchor0.05 | 118 / 16 | 5 | 0.63089 | 0.54698 |
| GRPO g4 anchor0.05 | 118 / 16 | 8 | 0.65769 | 0.56923 |
| GRPO g4 anchor0.05 | 118 / 16 | 10 | 0.68475 | 0.58932 |

结论：

- 当前 clean-val best 更新为 `label_RC`、M0 初始化、原始 GRPO、
  `group_size=4`、`supervised_anchor=0.05`、118 train / 16 val、10 epoch。
- 10 epoch reload Dice mean 0.58932，相对 8 epoch 的 0.56923 提升 +0.02009；
  相对 5 epoch full-train 的 0.54698 提升 +0.04234；
  相对 32-case GRPO 的 0.44198 提升 +0.14734；
  相对 supervised 32/16 的 0.42431 提升 +0.16501。
- 两 seed 的 best 分别在 epoch 10 / 9，说明 full-train 8 epoch 仍偏短；
  继续延长到 10 epoch 是有效的，不是单 seed 偶然。
- 这仍是 clean-val 结论；test fold 仍未用于这些 GRPO ablation。若上 test，
  必须先冻结配置。

A100 / 显存利用结论：

- 当前 decoder-only GRPO 单进程只占约 3.5-4GB 显存；瓶颈不是 A100 显存容量，
  而是单样本顺序 rollout / dataload / reference forward。
- 单卡并发两个训练可把瞬时 GPU util 提高到约 60%-70%，但会拖慢主实验；
  本轮 g6 与 e10 并发时 g6 明显慢，后续正式 ablation 优先单跑。
- `num_workers=2` 是低风险吞吐优化；后续正式 GRPO run 建议默认带上。
- batch>1 暂不安全；不要为了吃显存直接改 batch。

`group_size=6` A100 早停结果：

| Seed | Run | Stopped at | Best seen epoch | Val reward | Val Dice |
|---:|---|---:|---:|---:|---:|
| 2026 | `grpo/label_RC_m0init_fulltrain_16val_e8_g6_anchor005_seed2026` | epoch 4 early | 3 | 0.57176 | 0.51206 |

结论：g6 在前 3 epoch 明显落后于同条件 g4/e10（epoch3 Dice 0.51206 vs 0.53558），
且更慢；本轮作为效率负信号早停，不扩 seed，不作为正式 best 对比。

清理状态：

- e10 seed2026 / seed2027 两个训练目录均只保留 `sam_model_val_dice_best.pth`。
- g6 早停目录也只保留 `sam_model_val_dice_best.pth` 和日志/CSV。
- 已删除本轮重复的 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`。
- A100 当前空闲。

下一步建议：

1. 若继续 clean-val 内推进，优先小心跑 12 epoch seed2026（同配置、`num_workers=2`）；
   但由于 e10 已很强，若 seed2026 没超过 e10 reload，就不补 seed2027。
2. 若准备论文级 test，应先冻结配置为
   `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`，
   然后只做一次 held-out test 对比，不在 test 上调 epoch / reward / postprocess。
3. 新 idea 仍建议放在训练侧：boundary/surface reward 或 full image encoder low-lr smoke；
   不要回到 inference/postprocess 搜索。

---

## 28. 2026-07-06 insight 审计与跨领域 survey：算法是否真的 work

本节回答“当前算法到底能不能行、有没有真正 insight、之前尝试是否都不如原始 GRPO”。
已开两个只读子 agent：一个复核本地实验结果和日志，一个做外部论文/跨领域方法 survey。
两者均未改文件、未碰 test、未做 inference/postprocess 搜索。

硬结论：当前原始 GRPO 主线 work，而且不是微弱提升。所有数字均为同一
`label_RC`、patient-level split、clean-val、reload audit-only 口径：

| Method | Train / Val | Epochs | Reward mean | Dice mean |
|---|---:|---:|---:|---:|
| M0 init | 4 overfit init / 16 | - | 0.30775 | 0.29950 |
| Supervised best | 32 / 16 | 5 | 0.47593 | 0.42431 |
| GRPO g2 | 32 / 16 | 5 | 0.47262 | 0.42269 |
| GRPO g4 anchor0.05 | 32 / 16 | 5 | 0.49561 | 0.44198 |
| GRPO g4 anchor0.05 | 64 / 16 | 5 | 0.59016 | 0.51247 |
| GRPO g4 anchor0.05 | 96 / 16 | 5 | 0.61763 | 0.53777 |
| GRPO g4 anchor0.05 | 118 / 16 | 5 | 0.63089 | 0.54698 |
| GRPO g4 anchor0.05 | 118 / 16 | 8 | 0.65769 | 0.56923 |
| GRPO g4 anchor0.05 | 118 / 16 | 10 | 0.68475 | 0.58932 |

关键增益：

- 当时旧 scan-level split 内的 best：`GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`。
- 相对 supervised 32/16：Dice 0.58932 - 0.42431 = +0.16501。
- 相对 32-case GRPO：Dice 0.58932 - 0.44198 = +0.14734。
- 相对 118-train 5 epoch：Dice 0.58932 - 0.54698 = +0.04234。
- e10 两 seed reload audit Dice 分别为 0.59415 / 0.58450，不是单 seed 偶然。

真正 insight：

1. 有效的不是“随便 RL 一下”，而是 `group_size=4 + supervised_anchor=0.05 + KL + Dice/volume/component dense reward` 这个组合。
   g2 没超过 supervised，g4 后超过；扩 train cases 从 32 -> 64 -> 96 -> 118 基本单调上升；
   full-train 从 5 -> 8 -> 10 epoch 继续上升，说明此前主要欠训练/欠样本。
2. 原样 pairwise/DPO-style 方向在当前 dense segmentation 上是负结果。
   pure pairwise Dice mean 0.41865，低于 supervised 0.42431 和原始 GRPO32 0.44198；
   GRPO+pairwise 也只有 0.42739。原因推断：best-vs-worst rank loss 丢掉了 voxel-level dense reward magnitude，
   还会把低质量整幅 mask 当反例放大噪声。
3. center-of-mass reward 也是负结果。Dice mean 0.42915，略高 supervised 但低于原始 GRPO32 0.44198。
   center 奖励太粗，只约束位置，不直接优化 overlap、边界和局部形状；reward scale 也不可和旧 reward 直接横比。
4. A100 显存不是当前瓶颈。decoder-only GRPO 只占约 3.5-4GB，瓶颈是单样本顺序 rollout/reference forward/dataload。
   g6 seed2026 到 epoch3 Dice 0.51206，慢且落后于 g4/e10 同 epoch 0.53558；不要为了吃显存盲目加 group size。
5. 目前最有价值的科研主线仍是训练侧：延长原始 GRPO、边界/表面奖励、边界不确定性 anchor、hard-case curriculum、
   轻量 encoder 适配。不要回到 test-time postprocess 搜索。

外部 survey 对当前代码的可迁移点：

| Source | 对当前项目的启发 | 当前决策 |
|---|---|---|
| DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300 | GRPO 用 group-relative advantage 和 KL，省掉 value model，适合有规则 reward 的任务。 | 支持继续保留原始 GRPO 框架，而不是退回纯 supervised。 |
| DINO-R1 / GRQO: https://arxiv.org/abs/2505.24025 | 视觉 foundation model 也可以用 group-normalized query reward + KL 做 RL-style alignment。 | 支持“group-relative reward 可以迁移到视觉/分割”，但要设计 dense reward。 |
| SAMPO: https://arxiv.org/abs/2508.02464 | SAM-style segmentation 可做 preference optimization，并在 medical segmentation 场景验证。 | 方向上支持偏好/对齐训练；但本地 naive pairwise 已负，不能直接扩 pairwise。 |
| DPO: https://arxiv.org/abs/2305.18290 | DPO 在 LLM preference 中简单稳定。 | 当前 dense mask 任务中 rank-only pairwise 丢信息，作为负结果保留，不继续烧卡。 |
| Boundary Loss: https://arxiv.org/abs/1812.07032 | boundary/surface loss 可补 Dice/CE 对不平衡分割和边界的不足。 | 下一批新 idea 首选 surface Dice / boundary-F1 reward，而不是 center reward。 |
| Boundary Uncertainty: https://arxiv.org/abs/2111.00533 | 医学图像边界标注有不确定性，可只在边界带做 soft/降权处理。 | 可做 `anchor-mode=boundary_weighted`，降低 anchor 对噪声边界的硬约束。 |
| Dual uncertainty self-training: https://arxiv.org/abs/2304.04441 | sample-level / pixel-level uncertainty 可用于 curriculum 和伪标签降噪。 | 可迁移为 hard-case sampler 或 uncertainty-weighted anchor，不需要先引入 unlabeled test。 |

代码插点：

- reward：`scripts/train_sammed3d_grpo.py::segmentation_reward()`。
- rollout/group diversity：`scripts/train_sammed3d_grpo.py::rollout_group()`。
- 训练循环/日志：`scripts/train_sammed3d_grpo.py::train()`。
- 可训练范围：`scripts/train_sammed3d_brats.py::configure_trainable()`。

下一步优先级：

1. P0：如果要上 test，先冻结当前配置
   `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`；
   test 只跑一次，不在 test 上调 epoch/reward/postprocess。
2. P1：若继续 clean-val 内科研，先跑同配置 12 epoch seed2026。
   只有 reload audit 超过 e10 seed2026 Dice 0.59415，才补 seed2027；否则不继续延长。
3. P2：实现小成本 boundary/surface reward，先 32/16 e5 seed2026 smoke；
   Dice 必须超过旧 GRPO32 mean 0.44198，再补 seed2027 / 扩 full train。
4. P3：实现边界不确定性 supervised anchor 或 hard-case curriculum，仍先 32/16 小实验。
5. P4：若 decoder-only 到平台，再做 image encoder last-block/LoRA low-lr smoke；
   不要直接全量 encoder 大改。

当前不建议继续：pure pairwise、`grpo_pairwise coef0.05 beta8`、center reward coef0.05、
g6、batch>1、任何 test-time inference/postprocess 搜索。

---

## 29. 2026-07-06 e12 seed2026 复核：训练内略高，reload audit 未超过 e10

目的：按第 28 节 P1，只做同配置 12 epoch seed2026 复核，判断 e10 是否仍欠训练。
仍固定 `label_RC`、M0 初始化、原始 `grpo` objective、118 train / 16 val、同一
patient-level split、同一 clean-val、同一 reload audit-only 口径；test fold 未使用。

设置：

- Run：`workspace/BraTS/grpo/label_RC_m0init_fulltrain_16val_e12_g4_anchor005_nw2_seed2026`
- Audit：`workspace/BraTS/grpo/audit_label_RC_16val_grpo118_e12_g4_anchor005_nw2_best_seed2026`
- `group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`
- `num_clicks=5`，`num_workers=2`，`seed=2026`，单 A100

训练内 best：

| Seed | Best epoch | Val reward | Val Dice |
|---:|---:|---:|---:|
| 2026 | 10 | 0.69336 | 0.59470 |

训练内轨迹后半程：

| Epoch | Train Dice | Val reward | Val Dice |
|---:|---:|---:|---:|
| 8 | 0.67792 | 0.67466 | 0.58081 |
| 9 | 0.69454 | 0.66959 | 0.58020 |
| 10 | 0.69363 | 0.69336 | 0.59470 |
| 11 | 0.71020 | 0.66589 | 0.57971 |
| 12 | 0.72162 | 0.67051 | 0.58163 |

reload audit-only：

| Seed | Reward | Dice |
|---:|---:|---:|
| 2026 | 0.68425 | 0.58548 |

公平对比：

| Run | Seed | Training best Dice | Reload Dice |
|---|---:|---:|---:|
| e10 g4 anchor0.05 nw2 | 2026 | 0.59192 | 0.59415 |
| e12 g4 anchor0.05 nw2 | 2026 | 0.59470 | 0.58548 |

结论：

- e12 seed2026 训练内 best 略高于旧 e10 seed2026，但 reload audit 明显低于 e10
  reload（0.58548 vs 0.59415）。
- 因此不补 seed2027；当前 clean-val 主配置仍冻结为
  `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`。
- 这给出新 insight：继续延长到 12 epoch 的收益不稳，clean-val 训练内 best 不能替代
  保存 checkpoint 后的 reload audit；后续不要继续盲目拉长 epoch。
- 下一步若不上 test，应转向训练侧新 idea 的小成本 32/16 smoke：
  boundary/surface reward 或 boundary-weighted anchor，而不是继续 e14/e16。

清理状态：

- e12 训练目录已删除 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`。
- e12 训练目录只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 30. 2026-07-06 boundary/surface reward 小实验：coef0.05/tol2 判负

目的：按第 28/29 节建议，尝试训练侧 boundary/surface reward，替代此前过粗的
center-of-mass reward。仍只做小成本 32 train / 16 val seed2026；若不过原始 GRPO32
门槛，则不扩 seed2027 / full train。test fold 未使用。

代码变更：

- 文件：`scripts/train_sammed3d_grpo.py`
- 新增默认关闭参数：`--surface-reward-coef`（默认 0.0）和 `--surface-tolerance`（默认 2.0）。
- 新增 `surface_dice_reward(pred, gt, tolerance)`：
  - 用 `scipy.ndimage.binary_erosion` 提取 pred/GT surface；
  - 用 `distance_transform_edt` 计算容差内 surface match；
  - 返回 symmetric surface Dice。
- 当 `surface_reward_coef > 0` 时，reward 额外加
  `surface_reward_coef * surface_dice_reward`。
- `train_rollouts.csv` / `val_reward_audit.csv` / `reward_audit_summary.json`
  记录 `surface_reward`；默认 0 时不改变旧 GRPO 配置行为。

设置：

- Run：`workspace/BraTS/grpo/label_RC_m0init_32train_16val_e5_g4_anchor005_surface005_tol2_seed2026`
- Audit：`workspace/BraTS/grpo/audit_label_RC_16val_grpo32_g4_anchor005_surface005_tol2_best_seed2026`
- target：`label_RC`
- 初始化：M0 checkpoint `workspace/BraTS/training/m0_label_RC_overfit4/sam_model_val_dice_best.pth`
- train / val：32 / 16，同一 patient-level split，同一 clean-val
- objective：原始 `grpo`
- `group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`
- `surface_reward_coef=0.05`，`surface_tolerance=2`
- `num_clicks=5`，`num_workers=2`，`seed=2026`，单 A100

训练内结果：

| Epoch | Train Dice | Train surface | Val reward | Val Dice | Val surface |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.35158 | 0.47623 | 0.44426 | 0.38480 | 0.53084 |
| 2 | 0.44160 | 0.57610 | 0.47619 | 0.40763 | 0.53453 |
| 3 | 0.51202 | 0.66734 | 0.47938 | 0.40717 | 0.53413 |
| 4 | 0.54498 | 0.71776 | 0.49576 | 0.42020 | 0.55416 |
| 5 | 0.57130 | 0.74758 | 0.50386 | 0.42478 | 0.55946 |

Best epoch：epoch 5，val reward 0.50386，val Dice 0.42478。

reload audit-only：

| Seed | Reward | Dice | Surface reward |
|---:|---:|---:|---:|
| 2026 | 0.50294 | 0.42378 | 0.56455 |

公平对比：

| Method | Train / Val | Seed 2026 reload Dice | Two-seed mean if available |
|---|---:|---:|---:|
| Supervised best | 32 / 16 | 0.42186 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 | 0.44198 |
| GRPO + surface coef0.05 tol2 | 32 / 16 | 0.42378 | - |

结论：

- surface reward coef0.05/tol2 没超过原始 GRPO32，甚至只接近 supervised。
- 训练集 Dice 和 surface reward 明显上升，但 clean-val Dice 增长慢，说明这个 surface 项在当前形式下
  更像训练集形状拟合/边界贴合，不改善泛化 overlap。
- 速度也很差：surface reward 每个 rollout 都要做 3D distance transform，32 train 一个 epoch
  约 113 秒；相对原始 GRPO 吞吐不划算。
- 不补 seed2027，不扩 118 train；当前 clean-val 主配置仍是
  `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`。
- 若继续 boundary 方向，不建议继续 reward 加项；更合理的下一步是
  `boundary_weighted` supervised anchor（边界带降权/soft label），先 32/16 smoke。

清理状态：

- surface 训练目录已删除 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`。
- surface 训练目录只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 31. 2026-07-06 boundary-weighted anchor 小实验：判负

目的：第 30 节 surface reward 作为 reward 加项判负后，继续验证另一种 boundary 思路：
不改变 GRPO reward，而是把 supervised anchor 从硬 DiceCE 改成可选
`boundary_weighted` anchor，对 GT 边界带降权，降低边界标注噪声对 anchor 的硬约束。
仍只做 32 train / 16 val seed2026；不过原始 GRPO32 则不扩 seed。test fold 未使用。

代码变更：

- 文件：`scripts/train_sammed3d_grpo.py`
- 新增默认关闭参数：
  - `--anchor-mode dicece|boundary_weighted`，默认 `dicece`
  - `--boundary-ignore-radius`，默认 1
  - `--boundary-weight`，默认 0.5
- `dicece` 模式保持旧 `DiceCELoss(sigmoid=True, squared_pred=True)` 行为。
- `boundary_weighted` 模式：
  - 用 3D max-pool dilation / erosion 从 GT 生成 boundary band；
  - boundary band 权重设为 `boundary_weight`，核心前景和明确背景权重为 1；
  - anchor loss 为 `0.5 * weighted BCE + 0.5 * weighted soft Dice`。

设置：

- Run：`workspace/BraTS/grpo/label_RC_m0init_32train_16val_e5_g4_anchor005_boundaryw05_r1_seed2026`
- Audit：`workspace/BraTS/grpo/audit_label_RC_16val_grpo32_g4_anchor005_boundaryw05_r1_best_seed2026`
- target：`label_RC`
- 初始化：M0 checkpoint `workspace/BraTS/training/m0_label_RC_overfit4/sam_model_val_dice_best.pth`
- train / val：32 / 16，同一 patient-level split，同一 clean-val
- objective：原始 `grpo`
- `group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`
- `anchor_mode=boundary_weighted`，`boundary_ignore_radius=1`，`boundary_weight=0.5`
- `num_clicks=5`，`num_workers=2`，`seed=2026`，单 A100

训练内结果：

| Epoch | Train Dice | Val reward | Val Dice |
|---:|---:|---:|---:|
| 1 | 0.34478 | 0.39624 | 0.36287 |
| 2 | 0.43220 | 0.43980 | 0.39486 |
| 3 | 0.50444 | 0.40581 | 0.37194 |
| 4 | 0.53086 | 0.44730 | 0.40334 |
| 5 | 0.56629 | 0.44416 | 0.39986 |

Best epoch：epoch 4，val reward 0.44730，val Dice 0.40334。

reload audit-only：

| Seed | Reward | Dice |
|---:|---:|---:|
| 2026 | 0.45605 | 0.40737 |

公平对比：

| Method | Train / Val | Seed 2026 reload Dice | Two-seed mean if available |
|---|---:|---:|---:|
| Supervised best | 32 / 16 | 0.42186 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 | 0.44198 |
| GRPO + surface coef0.05 tol2 | 32 / 16 | 0.42378 | - |
| GRPO + boundary-weighted anchor | 32 / 16 | 0.40737 | - |

结论：

- boundary-weighted anchor 明显不如原始 DiceCE anchor；甚至低于 supervised baseline。
- 训练 Dice 从 0.34478 增至 0.56629，但 val Dice 峰值只有 0.40334，说明它加剧了小样本过拟合，
  没有改善 clean-val 泛化。
- 该方向不补 seed2027，不扩 full train。原始 `DiceCELoss` anchor 保持为当前主线默认。
- boundary 系列本轮结论：直接 surface reward 加项和 boundary-weighted anchor 都不如原始 GRPO。
  后续不要继续沿 boundary 系数/radius 做网格搜索。

清理状态：

- boundary-weighted 训练目录已删除 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`。
- boundary-weighted 训练目录只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 32. 2026-07-06 hard-case curriculum 小实验：判负

目的：按第 28/31 节建议，验证 hard-case curriculum 是否能把训练梯度集中到低 reward 病例。
仍只做 32 train / 16 val seed2026；不过原始 GRPO32 则不扩 seed2027 / full train。test fold 未使用。

代码变更：

- 文件：`scripts/train_sammed3d_grpo.py`
- 新增默认关闭参数：
  - `--curriculum-mode none|hard_case`，默认 `none`
  - `--hard-case-after-epoch`，默认 2
  - `--hard-case-fraction`，默认 0.30
  - `--hard-case-weight`，默认 2.0
  - `--hard-case-std-coef`，默认 0.0
- `hard_case` 模式每个 epoch 结束后按 case-level mean reward 排序，最低 fraction 的病例下一轮
  weight 设为 `hard_case_weight`，通过 `WeightedRandomSampler(replacement=True)` 重采样。
- 额外写 `curriculum_cases.csv`，记录每个 epoch 的 case score / weight / is_hard。
- 默认 `none` 不改变原始 GRPO 行为。
- 备注：本次 run 的 `history.csv` 中 epoch2 的 `hard_case_count` 记录的是已计算出的下一轮 hard
  cases；实际加权从 epoch3 开始。代码已在 run 后修正，后续 history 只记录实际启用的 hard-case
  采样。

设置：

- Run：`workspace/BraTS/grpo/label_RC_m0init_32train_16val_e5_g4_anchor005_hardcurr30_w2_after2_seed2026`
- Audit：`workspace/BraTS/grpo/audit_label_RC_16val_grpo32_g4_anchor005_hardcurr30_w2_after2_best_seed2026`
- target：`label_RC`
- 初始化：M0 checkpoint `workspace/BraTS/training/m0_label_RC_overfit4/sam_model_val_dice_best.pth`
- train / val：32 / 16，同一 patient-level split，同一 clean-val
- objective：原始 `grpo`
- `group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`
- `curriculum_mode=hard_case`，`hard_case_after_epoch=2`，`hard_case_fraction=0.30`，
  `hard_case_weight=2.0`，`hard_case_std_coef=0.0`
- `num_clicks=5`，`num_workers=2`，`seed=2026`，单 A100

训练内结果：

| Epoch | Active hard cases | Next hard cases | Train Dice | Val reward | Val Dice |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 10 | 0.35114 | 0.41440 | 0.38153 |
| 2 | 0* | 10 | 0.44323 | 0.44848 | 0.40540 |
| 3 | 10 | 6 | 0.47340 | 0.44748 | 0.40060 |
| 4 | 6 | 6 | 0.47994 | 0.44467 | 0.40358 |
| 5 | 6 | 7 | 0.56575 | 0.47613 | 0.42745 |

`*` 本次 run 的 CSV 记录为 10，但实际加权从 epoch3 开始；代码已修正。

Best epoch：epoch 5，val reward 0.47613，val Dice 0.42745。

reload audit-only：

| Seed | Reward | Dice |
|---:|---:|---:|
| 2026 | 0.47023 | 0.42267 |

公平对比：

| Method | Train / Val | Seed 2026 reload Dice | Two-seed mean if available |
|---|---:|---:|---:|
| Supervised best | 32 / 16 | 0.42186 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 | 0.44198 |
| GRPO + hard-case curriculum | 32 / 16 | 0.42267 | - |

结论：

- hard-case curriculum 这个配置没有超过原始 GRPO32，reload Dice 只接近 supervised。
- 加权后训练 Dice 到 epoch5 升到 0.56575，但 val Dice 只有 0.42745；说明对低 reward train case
  重采样更像加强小样本拟合，没有提升 clean-val 泛化。
- 不补 seed2027，不扩 full train；也不建议继续做 hard-case fraction/weight 网格搜索。
- 当前 clean-val 主配置仍是
  `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`。
- 负结果后的下一步更应转向提升 rollout 多样性本身，例如 `logit_noise` 或 decoder dropout，
  而不是继续重排同一批训练样本。

清理状态：

- hard-case curriculum 训练目录已删除 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`。
- hard-case curriculum 训练目录只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 33. 2026-07-06 rollout logit noise 小实验：接近但未超主线

目的：第 32 节 hard-case curriculum 判负后，按建议验证“提升 rollout 多样性本身”是否能改善
GRPO。当前 GRPO 的组内多样性主要来自不同 seed 的 click trajectory；本实验在训练阶段对
rollout logits 加受控高斯噪声，让 reward/action 看到更丰富的近阈值候选 mask。validation 和
reload audit 仍使用确定性 logits，不加噪声。仍只做 32 train / 16 val seed2026；test fold 未使用。

代码变更：

- 文件：`scripts/train_sammed3d_grpo.py`
- 新增默认关闭参数：
  - `--rollout-logit-noise-std`，默认 `0.0`
- 当 `enable_rollout_noise=True` 且该参数大于 0 时：
  - `scored_logits = logits + randn_like(logits) * rollout_logit_noise_std`
  - reward 和 threshold action 使用 `scored_logits`
  - policy logprob / KL / supervised anchor 仍使用原始 `logits`
- `run_audit()` 默认不启用 rollout noise，因此训练内 val 和 reload audit 口径保持确定性。
- 默认 `0.0` 不改变原始 GRPO 复现行为。

设置：

- target：`label_RC`
- 初始化：M0 checkpoint `workspace/BraTS/training/m0_label_RC_overfit4/sam_model_val_dice_best.pth`
- train / val：32 / 16，同一 patient-level split，同一 clean-val
- objective：原始 `grpo`
- `group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`
- `rollout_logit_noise_std=0.25` 和 `0.50`
- `num_clicks=5`，`num_workers=2`，`seed=2026`，单 A100
- 两个 run 并发执行；A100 util 约 97%，总显存占用约 7GB。

训练内结果：

| Noise std | Best epoch | Train Dice @ best | Val reward | Val Dice |
|---:|---:|---:|---:|---:|
| 0.25 | 4 | 0.54920 | 0.48496 | 0.43208 |
| 0.50 | 4 | 0.54599 | 0.48951 | 0.43720 |

reload audit-only：

| Noise std | Audit dir | Reward | Dice |
|---:|---|---:|---:|
| 0.25 | `grpo/audit_label_RC_16val_grpo32_g4_anchor005_logitnoise025_best_seed2026` | 0.48539 | 0.43340 |
| 0.50 | `grpo/audit_label_RC_16val_grpo32_g4_anchor005_logitnoise050_best_seed2026` | 0.49483 | 0.44108 |

公平对比：

| Method | Train / Val | Seed 2026 reload Dice | Two-seed mean if available |
|---|---:|---:|---:|
| Supervised best | 32 / 16 | 0.42186 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 | 0.44198 |
| GRPO + logit noise std0.25 | 32 / 16 | 0.43340 | - |
| GRPO + logit noise std0.50 | 32 / 16 | 0.44108 | - |

结论：

- `std=0.50` 接近原始 GRPO32 的两 seed mean，但低于同 seed 原始 GRPO32 reload Dice
  0.44541，未形成独立增益。
- `std=0.25` 更弱，说明轻噪声没有明显改善候选多样性。
- 噪声组训练 Dice 能继续升高，但 epoch5 val Dice 回落；这更像训练扰动/正则化，而不是稳定泛化提升。
- 不补 seed2027，不扩 full train；当前 clean-val 主配置仍是
  `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`。
- 后续不建议继续做 logit-noise std 网格。若继续“多样性”方向，应考虑更结构化的多样性：
  例如 click trajectory 级别的 controlled perturbation 或 uncertainty-aware anchor，而不是直接扰动 logits。

清理状态：

- 两个 logit-noise 训练目录已删除 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`。
- 两个训练目录只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 34. 2026-07-06 tumor_ET 跨目标迁移：seed2026 小正，seed2027 反转

目的：前面几乎所有 GRPO 主结论都来自 `label_RC`。为了确认算法是否只对 RC 有效，本轮把当前
主配置 `group_size=4 + supervised_anchor=0.05` 迁移到另一个弱目标 `tumor_ET`。仍使用
patient-level split、clean-val、reload audit-only；test fold 未使用。

数据规模：

| Target | Train cases | Val cases | Test cases |
|---|---:|---:|---:|
| `tumor_ET` | 102 | 14 | 29 |

先补 `tumor_ET` M0 初始化：

- Run：`workspace/BraTS/training/m0_tumor_ET_overfit4`
- checkpoint：SAM-Med3D-turbo 原始 checkpoint
- `max_train_cases=4`，`overfit=True`，20 epoch，`lr=5e-4`
- best overfit-val Dice：0.54356（epoch 20）
- clean-val audit：

| Method | Audit dir | Reward | Dice |
|---|---|---:|---:|
| ET M0 init | `grpo/audit_tumor_ET_14val_m0init_g4_seed2026` | 0.05785 | 0.13207 |

说明：M0 overfit4 只是 target-specific warm start / sanity，不是泛化模型；clean-val 起点仍很低。

32/14 supervised vs GRPO 设置：

- 初始化：`workspace/BraTS/training/m0_tumor_ET_overfit4/sam_model_val_dice_best.pth`
- train / val：32 / 14，同一 patient-level split
- supervised：`DiceCELoss` decoder-only，5 epoch，`lr=2e-5`
- GRPO：原始 `grpo` objective，`group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`，
  5 epoch，`lr=2e-5`
- seeds：2026、2027

训练内 best：

| Seed | Method | Best epoch | Val reward | Val Dice |
|---:|---|---:|---:|---:|
| 2026 | supervised | 4 | - | 0.38165 |
| 2026 | GRPO g4 anchor0.05 | 5 | 0.42340 | 0.39713 |
| 2027 | supervised | 5 | - | 0.39739 |
| 2027 | GRPO g4 anchor0.05 | 4 | 0.40687 | 0.38348 |

reload audit-only：

| Seed | Method | Audit dir | Reward | Dice |
|---:|---|---|---:|---:|
| 2026 | supervised | `grpo/audit_tumor_ET_14val_sup32_e5_best_g4_seed2026` | 0.40728 | 0.38604 |
| 2026 | GRPO g4 anchor0.05 | `grpo/audit_tumor_ET_14val_grpo32_g4_anchor005_best_seed2026` | 0.41359 | 0.39092 |
| 2027 | supervised | `grpo/audit_tumor_ET_14val_sup32_e5_best_g4_seed2027` | 0.42739 | 0.40026 |
| 2027 | GRPO g4 anchor0.05 | `grpo/audit_tumor_ET_14val_grpo32_g4_anchor005_best_seed2027` | 0.39611 | 0.38089 |

两 seed mean：

| Method | Train / Val | Dice mean |
|---|---:|---:|
| ET supervised | 32 / 14 | 0.39315 |
| ET GRPO g4 anchor0.05 | 32 / 14 | 0.38590 |

结论：

- `tumor_ET` 上 GRPO 不是稳定正结果：seed2026 小幅超过 supervised（+0.00487），但 seed2027
  明显低于 supervised（-0.01937）。
- 两 seed mean 下 GRPO 比 supervised 低 -0.00725，因此不能把 RC 的强结论直接外推到 ET。
- 这说明当前 GRPO reward/anchor 对 ET 这种更小、更稀疏的增强肿瘤区域不够稳，可能更受
  click trajectory 和样本内方差影响。
- 不扩 ET 到 64/full train，不上 test；若继续 ET，需要先重审 ET 的 reward/anchor 或更换
  target-specific 训练策略，而不是照搬 RC 配置。
- 当前主线结论保持：`label_RC` 上 `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch`
  仍是最强 clean-val 结果；ET 暂时作为跨目标负/混合结果记录。

清理状态：

- ET M0、ET supervised 两 seed、ET GRPO 两 seed 的训练目录均已删除重复
  `sam_model_latest.pth` / `sam_model_val_loss_best.pth` / `sam_model_val_reward_best.pth`。
- 上述训练目录均只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 35. 2026-07-06 tumor_TC 跨目标迁移：同样不是稳定正结果

目的：在 `tumor_ET` 迁移不稳定之后，再用另一个弱目标 `tumor_TC` 复核当前
`GRPO g4 anchor0.05` 是否能跨目标泛化。仍固定 patient-level split、clean-val、
reload audit-only；test fold 未使用。

数据规模：

| Target | Train cases | Val cases | Test cases |
|---|---:|---:|---:|
| `tumor_TC` | 103 | 14 | 29 |

先补 `tumor_TC` M0 初始化：

- Run：`workspace/BraTS/training/m0_tumor_TC_overfit4`
- checkpoint：SAM-Med3D-turbo 原始 checkpoint
- `max_train_cases=4`，`overfit=True`，20 epoch，`lr=5e-4`
- best overfit-val Dice：0.52325（epoch 20）
- clean-val audit：

| Method | Audit dir | Reward | Dice |
|---|---|---:|---:|
| TC M0 init | `grpo/audit_tumor_TC_14val_m0init_g4_seed2026` | 0.03469 | 0.11818 |

说明：和 ET 一样，M0 overfit4 能证明 decoder 可被目标拉动，但换到完整 clean-val 起点仍很低。

32/14 supervised vs GRPO 设置：

- 初始化：`workspace/BraTS/training/m0_tumor_TC_overfit4/sam_model_val_dice_best.pth`
- train / val：32 / 14，同一 patient-level split
- supervised：`DiceCELoss` decoder-only，5 epoch，`lr=2e-5`
- GRPO：原始 `grpo` objective，`group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`，
  5 epoch，`lr=2e-5`
- seeds：2026、2027
- A100 执行：四个独立进程并发跑在单张 A100 上，峰值约 10.9GB/80GB，GPU util 约 90%+；
  脚本仍是单进程单卡，不做 DDP。

训练内 best：

| Seed | Method | Best epoch | Val reward | Val Dice |
|---:|---|---:|---:|---:|
| 2026 | supervised | 4 | - | 0.41882 |
| 2026 | GRPO g4 anchor0.05 | 5 | 0.44210 | 0.41476 |
| 2027 | supervised | 5 | - | 0.42756 |
| 2027 | GRPO g4 anchor0.05 | 4 | 0.41929 (reward-best epoch 3) | 0.39969 |

reload audit-only：

| Seed | Method | Audit dir | Reward | Dice |
|---:|---|---|---:|---:|
| 2026 | supervised | `grpo/audit_tumor_TC_14val_sup32_e5_best_g4_seed2026` | 0.43951 | 0.41400 |
| 2026 | GRPO g4 anchor0.05 | `grpo/audit_tumor_TC_14val_grpo32_g4_anchor005_best_seed2026` | 0.44830 | 0.42027 |
| 2027 | supervised | `grpo/audit_tumor_TC_14val_sup32_e5_best_g4_seed2027` | 0.44819 | 0.42073 |
| 2027 | GRPO g4 anchor0.05 | `grpo/audit_tumor_TC_14val_grpo32_g4_anchor005_best_seed2027` | 0.42516 | 0.40505 |

两 seed mean：

| Method | Train / Val | Dice mean |
|---|---:|---:|
| TC supervised | 32 / 14 | 0.41736 |
| TC GRPO g4 anchor0.05 | 32 / 14 | 0.41266 |

结论：

- `tumor_TC` 和 `tumor_ET` 一样，GRPO 不是稳定正结果：seed2026 小幅超过 supervised
  （+0.00627），但 seed2027 明显低于 supervised（-0.01568）。
- 两 seed mean 下 GRPO 比 supervised 低 -0.00470，因此不能把 RC 上的强正结论直接外推到
  TC/ET。
- 不扩 TC 到 64/full train，不上 test。TC/ET 结果说明当前 `reward + anchor + group rollout`
  对不同目标的鲁棒性不足；下一步应先改训练算法，再回到 TC/ET 复核。
- 当前主线结论保持：`label_RC` 上 `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch`
  仍是最强 clean-val 结果。

双 agent sidecar 结论：

- 方法侧优先级：若跨目标不稳，优先做 `KL/anchor/reference schedule`、
  `variance-reduced advantage`、`mixed soft reward`；这些最贴近当前 GRPO 成功机制。
- 暂停继续投入：pairwise、center reward、surface reward、boundary-weighted anchor、
  hard-case curriculum、rollout logit-noise 网格、`g6`、单纯 e12/e14 延训。
- 性能侧优先级：A100 上继续并发独立实验；GRPO 不要强行 batch > 1 或 DDP。
  后续可低风险测试 `num_workers 0/2/4`、GRPO `.to(..., non_blocking=True)`、以及
  DataLoader `persistent_workers/prefetch_factor`，这些不改变科研口径。

建议下一步：

1. 在 `label_RC 32/16` 上先做小成本算法改造，而不是直接扩 TC/ET：
   - `supervised_anchor 0.08 -> 0.05 -> 0.03`
   - `kl_coef 0.04 -> 0.02 -> 0.01`
   - 或加入 leave-one-out / rank advantage 这类低成本 variance reduction。
2. 只有当新算法在 RC32 seed2026 超过原始 0.44541，并且 seed2027 不反转，再跑 full train 或回到
   TC/ET 复核。
3. test fold 仍未使用；除非先冻结最终配置，否则继续只用 clean-val。

清理状态：

- TC M0、TC supervised 两 seed、TC GRPO 两 seed 的训练目录均已删除重复
  `sam_model_latest.pth` / `sam_model_val_loss_best.pth` / `sam_model_val_reward_best.pth`。
- 上述训练目录均只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 36. 2026-07-06 RC anchor/KL schedule：seed2026 微正，seed2027 反转，不扩

目的：ET/TC 跨目标迁移显示原始 `g4+anchor0.05` 不够鲁棒后，先回到成功目标
`label_RC 32/16` 做低成本算法改造。测试更强早期约束、后期放开的 schedule：

- `supervised_anchor_schedule=0.08,0.05,0.03`
- `kl_coef_schedule=0.04,0.02,0.01`
- 5 epoch 展开为 anchor `[0.08,0.08,0.05,0.05,0.03]`，
  KL `[0.04,0.04,0.02,0.02,0.01]`

实现：

- `scripts/train_sammed3d_grpo.py` 新增默认关闭参数：
  `--supervised-anchor-schedule`、`--kl-coef-schedule`。
- 默认空字符串时行为与旧代码一致。
- `grpo_config.json` 会记录展开后的 per-epoch 权重；
  `history.csv` 每行记录当前 epoch 的 `supervised_anchor` 与 `kl_coef`。

训练内 best：

| Seed | Run | Best epoch | Val reward | Val Dice |
|---:|---|---:|---:|---:|
| 2026 | `grpo/label_RC_m0init_32train_16val_e5_g4_anchor_sched008005003_kl004002001_seed2026` | 4 | 0.49352 | 0.43979 |
| 2027 | `grpo/label_RC_m0init_32train_16val_e5_g4_anchor_sched008005003_kl004002001_seed2027` | 5 | 0.47622 | 0.42594 |

reload audit-only：

| Seed | Method | Audit dir | Reward | Dice |
|---:|---|---|---:|---:|
| 2026 | original GRPO g4 anchor0.05 | `grpo/audit_label_RC_16val_grpo32_g4_anchor005_best_g4_seed2026` | 0.50245 | 0.44541 |
| 2026 | anchor/KL schedule | `grpo/audit_label_RC_16val_grpo32_g4_anchor_sched008005003_kl004002001_best_seed2026` | 0.50406 | 0.44653 |
| 2027 | original GRPO g4 anchor0.05 | `grpo/audit_label_RC_16val_grpo32_g4_anchor005_best_g4_seed2027` | 0.48876 | 0.43855 |
| 2027 | anchor/KL schedule | `grpo/audit_label_RC_16val_grpo32_g4_anchor_sched008005003_kl004002001_best_seed2027` | 0.48003 | 0.42734 |

两 seed mean：

| Method | Train / Val | Dice mean |
|---|---:|---:|
| original GRPO g4 anchor0.05 | 32 / 16 | 0.44198 |
| anchor/KL schedule | 32 / 16 | 0.43694 |

结论：

- schedule 在 seed2026 只有极小提升（+0.00112），但 seed2027 明显反转（-0.01121）。
- 两 seed mean 下降 -0.00504，说明这不是稳定改进。
- 训练曲线也显示 seed2027 后期有 clean-val 回落，强早期约束再放开没有解决跨 seed 稳定性。
- 不扩 full train，不回到 TC/ET 复核，不上 test。
- 下一步若继续算法改造，应优先试更低成本的 advantage 方差控制（例如 rank advantage），或
  mixed soft reward；不要继续 schedule 网格。

清理状态：

- 两个 schedule 训练目录均已删除重复 `sam_model_latest.pth` 和
  `sam_model_val_reward_best.pth`。
- 上述训练目录均只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 37. 2026-07-06 RC rank advantage：未超过原始 GRPO，不补 seed2027

目的：在 anchor/KL schedule 两 seed 不稳定后，继续测试 sidecar 建议的低成本
variance-reduced advantage。原始 GRPO 用 reward 数值做 group-normalized advantage；
rank advantage 只使用 group 内排序，降低异常高/低 reward rollout 对梯度的支配。

实现：

- `scripts/train_sammed3d_grpo.py` 新增默认关闭参数
  `--advantage-mode standard|rank`。
- 默认 `standard` 与旧逻辑一致。
- `rank` 模式对同一 group 的 rollout 按 reward 排名，再把 rank 标准化为 advantage；
  不增加 forward，不改变 reward 定义，不改变 audit 口径。

设置：

- `label_RC`，M0 初始化，32 train / 16 val，5 epoch
- 原始 `grpo` objective，`group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`
- `advantage_mode=rank`
- seed2026 only，clean-val reload audit-only，不使用 test fold

结果：

| Method | Seed | Train best epoch | Train best val reward | Train best val Dice | Reload reward | Reload Dice |
|---|---:|---:|---:|---:|---:|---:|
| original GRPO g4 anchor0.05 | 2026 | - | - | - | 0.50245 | 0.44541 |
| rank advantage | 2026 | 4 | 0.48452 | 0.42980 | 0.50343 | 0.44234 |

结论：

- rank advantage 的 reload Dice 为 0.44234，低于原始同 seed 0.44541。
- 虽然 reload reward 与原始接近，但 Dice 没有提升，说明单纯把 reward 数值换成 rank 没解决泛化问题。
- 不补 seed2027，不扩 full train，不上 test。
- 当前已验证的两条“稳定性改造”（anchor/KL schedule、rank advantage）都没有超过原始 GRPO 两 seed 结论。
- 下一步如果继续算法侧，优先级应转向 `mixed soft reward` 或 reward-weighted BC/AWR，而不是继续 advantage/schedule 网格。

清理状态：

- rank advantage 训练目录已删除重复 `sam_model_latest.pth` 和
  `sam_model_val_reward_best.pth`。
- 训练目录只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 38. 2026-07-07 RC mixed-soft reward：seed2026 仅微正，不补 seed2027

目的：前面的 schedule 与 rank advantage 都没有稳定超过原始 GRPO 后，继续测试
更贴近医学分割的 reward shaping：保留 hard Dice 作为主 reward，但掺入少量 soft
Dice/soft IoU，让训练阶段看到概率边界和低置信度区域的信息。最终评估仍坚持 hard
mask reload audit-only。

实现：

- `scripts/train_sammed3d_grpo.py` 新增默认关闭参数：
  `--reward-mode hard|mixed_soft`、`--soft-reward-weight`。
- `mixed_soft` 的 overlap reward 为：
  `0.85 * hard_dice + 0.15 * (0.5 * soft_dice + 0.5 * soft_iou)`。
- `history.csv` / `reward_audit.csv` 额外记录 `soft_dice`、`soft_iou`、
  `soft_reward`、`overlap_reward`，方便判断 soft signal 与 hard audit 是否错配。
- `--audit-only` 已加护栏：standalone reload audit 强制使用 hard reward 口径，
  避免把 mixed-soft 参数带入最终对比。

设置：

- `label_RC`，M0 初始化，32 train / 16 val，5 epoch
- 原始 `grpo` objective，`group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`
- `reward_mode=mixed_soft`，`soft_reward_weight=0.15`
- seed2026 only，clean-val reload audit-only，不使用 test fold

结果：

| Method | Seed | Train best epoch | Train best val reward | Train best val Dice | Reload reward | Reload Dice |
|---|---:|---:|---:|---:|---:|---:|
| original GRPO g4 anchor0.05 | 2026 | - | - | - | 0.50245 | 0.44541 |
| mixed-soft reward w0.15 | 2026 | 4 | 0.46848 | 0.44163 | 0.50378 | 0.44743 |

结论：

- mixed-soft 的 hard reload Dice 为 0.44743，相对原始同 seed 0.44541 只提升
  +0.00203。
- 这个增益低于扩 seed 的最低阈值，可能属于 seed/selection 噪声；不补 seed2027，
  不扩 full train，不上 test。
- insight：soft overlap 确实能给训练提供更平滑信号，但当前 hard mask audit 才是最终指标；
  soft 指标和 hard Dice 存在目标错配，因此不能替代原始 hard Dice reward 主线。
- 当时旧 scan-level clean-val best 不变：
  `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`，
  两 seed reload Dice mean 0.58932。
- 下一步如果继续算法侧，比起继续调 soft weight，更值得试 reward-weighted BC/AWR
  或 reference-free/light-KL 的小筛选；仍必须先在 RC32 seed2026 过 0.44541 且有
  明确 margin，再补 seed2027。
- 文献对应关系：AWR（https://arxiv.org/abs/1910.00177）、AWAC
  （https://arxiv.org/abs/2006.09359）、CRR（https://arxiv.org/abs/2006.15134）
  都把 policy 更新写成 advantage/reward 加权的回归或最大似然；迁移到本任务时可理解为：
  在同一 group 内只用高 reward rollout 的 mask/click trajectory 对 decoder 做加权模仿，
  保留 dense reward magnitude，而不是只比较 best/worst 排名。

清理状态：

- mixed-soft 训练目录已删除重复 `sam_model_latest.pth` 和
  `sam_model_val_reward_best.pth`。
- 训练目录只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 39. 2026-07-07 RC GRPO+AWR / reward-weighted BC：三组 seed2026 均负

目的：mixed-soft 只给出 +0.002 的微弱信号后，继续测试文献启发的
reward-weighted BC / AWR 风格更新。思路来自 AWR/AWAC/CRR：在同一 group 内，
只对高 reward rollout 的 hard mask/click trajectory 做加权模仿，保留 dense reward
magnitude，避免 pairwise best-vs-worst 丢掉 reward 大小。

实现：

- `scripts/train_sammed3d_grpo.py` 新增默认关闭 objective：
  `--objective awr|grpo_awr`，默认仍是 `grpo`，旧实验不变。
- AWR 权重使用 group-relative standardized advantage，而不是 raw reward：
  `w_i = clamp(exp(A_i / temperature), max_weight)`；
  `--awr-positive-only` 时只模仿 `A_i > 0` 的 rollout。
- `grpo_awr` 保留原始 GRPO loss，再额外加
  `awr_coef * weighted_BC_loss`；`awr` 纯模式只作为诊断入口。
- CSV/history 额外记录 `advantage`、`awr_weight`、
  `train_awr_weight`、`val_awr_weight`，用于诊断权重是否过尖。
- reload audit 仍只用 `sam_model_val_dice_best.pth` 做 hard audit-only；
  AWR 参数不参与成功判定。

设置：

- `label_RC`，M0 初始化，32 train / 16 val，5 epoch
- `group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`
- seed2026 only，clean-val reload audit-only，不使用 test fold

结果：

| Method | AWR params | Train best epoch | Train best val Dice | Reload reward | Reload Dice |
|---|---|---:|---:|---:|---:|
| original GRPO g4 anchor0.05 | none | - | - | 0.50245 | 0.44541 |
| GRPO+AWR sharp | coef0.02, tau0.10, wmax5, positive-only | 4 | 0.43445 | 0.49257 | 0.43841 |
| GRPO+AWR sharp | coef0.05, tau0.10, wmax5, positive-only | 4 | 0.43196 | 0.49250 | 0.43593 |
| GRPO+AWR conservative | coef0.03, tau1.00, wmax3, positive-only | 4 | 0.43700 | 0.48751 | 0.43469 |

结论：

- 三组 AWR 都低于原始同 seed reload Dice 0.44541，且没有达到补 seed2027 的最低门槛。
- 尖锐权重版 `train_awr_weight` 均值约 2.5-2.9，可能过度模仿少数高 reward 自举 mask；
  保守版权重均值约 1.1-1.3，但仍不提升 hard Dice。
- insight：当前原始 GRPO 的正负 advantage 推拉已经比“只模仿高 reward rollout”
  更适合这个 dense mask 任务。AWR 容易把当前模型自己的局部好 mask 当伪标签强化，
  但没有提供比 Dice reward policy update 更好的泛化。
- 不补 seed2027，不扩 64/full train，不上 test。
- 当时旧 scan-level clean-val best 不变：
  `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`，
  两 seed reload Dice mean 0.58932。
- 下一步算法侧不要继续 AWR coef/temperature 网格。若继续探索，应优先考虑
  reference/KL 结构或数据/目标层面的泛化问题；但任何新方法仍需先在 RC32 seed2026
  明确超过 0.44541 且有足够 margin，再补 seed2027。

清理状态：

- 三个 AWR 训练目录均已删除重复 `sam_model_latest.pth` 和
  `sam_model_val_reward_best.pth`。
- 训练目录只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- test fold 仍未使用。

---

## 40. 2026-07-07 5-agent 审计与 RC full-train supervised clean 对照：GRPO 小幅优势仍成立

目的：用户要求开启多 agent 并行科研后，启动 5 个只读 side agents 做分工审计，
同时补一个关键缺口：此前 `label_RC` full-train GRPO 只和 32-case supervised 对比，
缺少 118-case 同规模 supervised control。这个对照决定“GRPO 是否真的优于监督训练”，
还是只是吃到了更多训练样本。

注意：旧 `audit_label_RC_16val_sup118_e10_best_g4_seed2026/2027` 数值不再作为结论；
本节使用 `_sup_clean_seed2026/2027` 完整 10 epoch 重跑后的 clean reload audit。

side-agent 分工：

- Data/Target Strategist：聚焦 RC 成功但 ET/TC 不稳，建议后续 multi-target 必须显式
  target-balanced，不能直接用混合 JSONL 加 `--max-train-cases` 截断。
- Algorithm Scout：建议优先做 supervised checkpoint -> local GRPO、target-conditioned
  prompt token、EMA reference KL；这些都不同于已失败的 AWR/mixed-soft/schedule/rank。
- Code Auditor：指出当前 `filter_rows()` 会按 `(target, case_id)` 排序后截断，
  因此 multi-target 小筛不能直接复用 `--max-train-cases`，否则 target sample 会偏。
- A100 Scheduler：建议小筛最多 2 个进程并发，full-train 单跑；但本轮 supervised
  对照显存占用低，两个 seed 并发 A100 util 到 99%。
- Results Auditor：指出最缺的统计对照就是 RC118 supervised full-train control。

设置：

- target：`label_RC`
- train / val：118 train / 16 val，同一 patient-level split、同一 clean-val
- 初始化：`workspace/BraTS/training/m0_label_RC_overfit4/sam_model_val_dice_best.pth`
- 方法：supervised decoder-only，DiceCE loss，5-click 协议，10 epoch，lr `2e-5`
- seeds：2026、2027
- 评估：训练后用 `sam_model_val_dice_best.pth` 做 GRPO audit-only hard reload；
  不使用 test fold

训练内 best：

| Seed | Best epoch | Train best val Dice |
|---:|---:|---:|
| 2026 | 8 | 0.58501 |
| 2027 | 9 | 0.58300 |

reload audit-only：

| Method | Seed | Audit dir | Reward | Dice |
|---|---:|---|---:|---:|
| supervised118 e10 clean | 2026 | `grpo/audit_label_RC_16val_fulltrain_e10_sup_clean_best_seed2026` | 0.65577 | 0.56454 |
| supervised118 e10 clean | 2027 | `grpo/audit_label_RC_16val_fulltrain_e10_sup_clean_best_seed2027` | 0.67153 | 0.58113 |
| GRPO118 e10 | 2026 | `grpo/audit_label_RC_16val_grpo118_e10_g4_anchor005_nw2_best_seed2026` | 0.69118 | 0.59415 |
| GRPO118 e10 | 2027 | `grpo/audit_label_RC_16val_grpo118_e10_g4_anchor005_nw2_best_seed2027` | 0.67832 | 0.58450 |

两 seed mean：

| Method | Train / Val | Epochs | Reload Dice mean |
|---|---:|---:|---:|
| supervised32 | 32 / 16 | 5 | 0.42431 |
| supervised118 clean | 118 / 16 | 10 | 0.57283 |
| GRPO118 g4 anchor0.05 | 118 / 16 | 10 | 0.58932 |

结论：

- full-train supervised 明显强于 32-case supervised，说明数据扩展本身贡献很大。
- 但同规模同 epoch 下，GRPO 仍比 supervised clean 高 +0.01649 Dice mean：
  seed2026 高 +0.02961，seed2027 高 +0.00337。
- 这没有达到此前临时设定的 +0.020 mean gate，且 seed2027 margin 很小；
  因此结论要写成“GRPO 有小幅同规模优势”，不要写成大幅领先。
- 更严谨的历史 insight：旧 scan-level split 内，`label_RC` 原始 GRPO 相对同规模
  supervised 有小幅优势；但该 split 存在纵向 subject 泄漏，不能写成 subject-disjoint
  泛化结论。full-train supervised 本身已很强，当时算法增益主要是最后一小段 hard Dice。
- 这个结论仍只是 clean-val 阶段；test fold 继续封存，直到最终配置冻结。

清理状态：

- 两个 supervised118 clean 训练目录已删除重复 `sam_model_latest.pth` 和
  `sam_model_val_loss_best.pth`。
- 两个目录均只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。
- 5 个 side-agent prompt/report 位于 `workspace/BraTS/agent_reports/`。

---

## 41. 2026-07-07 RC corrective GRPO：负结果，不扩 seed

目的：测试一个比 AWR 更保守的变体。正 advantage rollout 仍走原始 GRPO policy update；
负 advantage rollout 不再直接推 policy，而是加一个 supervised-anchor 风格 corrective loss，
希望把低 reward 采样拉回 GT 附近。

实现：

- `scripts/train_sammed3d_grpo.py` 新增默认关闭 objective：`--objective grpo_corrective`。
- `--corrective-coef` 控制负 advantage corrective loss 强度。
- reload audit 仍只用 `sam_model_val_dice_best.pth` 做 hard audit-only；
  corrective 参数不参与成功判定。

设置：

- `label_RC`，M0 初始化，32 train / 16 val，5 epoch
- `group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`
- seed2026 only，clean-val reload audit-only，不使用 test fold

结果：

| Method | Train best epoch | Train best val Dice | Reload reward | Reload Dice |
|---|---:|---:|---:|---:|
| original GRPO g4 anchor0.05 | - | - | 0.50245 | 0.44541 |
| corrective coef0.05 | 4 | 0.42664 | 0.47588 | 0.42572 |
| corrective coef0.10 | 4 | 0.43669 | 0.48941 | 0.43740 |

结论：

- 两个 corrective 设置都低于原始同 seed reload Dice 0.44541；
  coef0.10 比 coef0.05 好，但仍差 -0.00801 Dice。
- 不补 seed2027，不扩 full train，不上 test。
- insight：对负 advantage 只做 anchor correction 太保守，容易退回监督项；
  原始 GRPO 的正负 advantage 推拉仍是当前 RC 上更有效的更新方式。

清理状态：

- 两个 corrective 训练目录均已删除重复 `sam_model_latest.pth` 和
  `sam_model_val_reward_best.pth`。
- 两个目录均只保留 `sam_model_val_dice_best.pth` 和日志/CSV/JSON。

---

## 42. 2026-07-17：B200 + BraTS2024 全量 subject-clean work #2 重启

本节只记录 **work #2（训练/微调 SAM-Med3D 权重）**。本轮没有运行或修改
threshold / TTA / postprocess 搜索；sealed test 没有做任何 evaluation。

### 42.1 数据身份纠正与全量预处理

- `/mnt/bn/liveaim/songjunyi/brats2024` 有 1350 scans，但只有 **613 个 subjects**；
  `BraTS-GLI-xxxxx-100/-101/...` 是同一 subject 的纵向 timepoints。
- 历史 200 scans 只来自 39 subjects。旧 split 按完整 `case_id` 分组，train-val 重叠
  16/16 个 val subjects、train-test 重叠 27/27 个 test subjects。因此第 40 节的 RC
  `0.58932` 只保留为历史 scan-level/纵向泄漏结果，不能作为新协议晋级线。
- 新 schema v4 输出：
  `/mnt/bn/liveaim/songjunyi/brats2024_sammed3d_aflow_v4`，共 1350 scans / 613 subjects /
  6334 target rows；manifest SHA-256：
  `4d5cec4d66018789f35f39b9b2354c0910d6d01cdfbf345330f81d144186a65d`。
- target rows：NETC 547、RC 1143、SNFH 1350、ET 969、TC 975、WT 1350。
- 默认 `tumor_WT` 继续使用历史兼容语义 `{1,2,3,4}`；官方术后 WT `{1,2,3}`
  只能另名为 `tumor_WT_official`，没有静默换标签。
- 深度 audit 检查了 source 不变、shape/affine/dtype、binary mask 和 foreground 一致性；
  报告：`data/datasets/brats2024_full_audit_v4.json`。

### 42.2 新 subject-disjoint split 与 test seal

- split：`data/datasets/brats2024_subject_split_seed2026_v4`。
- 419 / 67 / 127 train/val/test subjects，对应 948 / 133 / 269 scans，target rows 为
  4431 / 630 / 1273；subject overlap 为 0。
- 历史 39 subjects 与 4 个曾做人工/原始体素检查的 subjects 全部强制进入 train。
- split 创建前，自动 preprocessing invariant QA 抽样了 10 subjects，其中 3 个后来随机
  进入 test；该 QA 只检查 shape/affine/dtype/binary/foreground，不含模型预测、指标或
  方法选择。seal 建立后至最终协议冻结前没有读取 test JSONL/cases，也没有 test evaluation；
  冻结后的唯一尝试已访问并因环境错误失败，见 42.9。
- split manifest SHA-256：
  `43e37345150218698a4df6349c266e5177641db0ebb5e2542e40a5ade01be27b`。
- test JSONL 只存在于 `sealed_test/`；checkpoint audit 会硬拒绝 resolved path 中包含
  `sealed_test` 的输入或输出。最终配置冻结前继续禁止 test evaluation。

### 42.3 B200 适配结论

- 两张卡均为 NVIDIA B200 183359 MiB，SM100。原 venv 的 torch 2.7.1+cu126 不含
  SM100 kernel，不能直接用于 B200。
- 已用 `scripts/bootstrap_b200_env.sh` 构建 overlay：torch `2.12.0+cu130`、
  torchvision `0.27.0`，launcher 为 `/tmp/aflow-sammed3d-b200-python`。
- overlay 必须让自身 cuDNN 9.20 优先，同时保留 worker 的 CUDA forward-compat 路径；
  `scripts/run_worker_research.sh` 通过 `AFLOW_B200_SITE` 完成该设置。
- 最终 probe 已通过：CUDA available、B200 capability `(10,0)`、普通 CUDA kernel、
  cuDNN 92000、Conv3D finite 全部正常。
- 真实 SAM-Med3D BF16 smoke：decoder-only、target token、12-block Q/V LoRA r4/a8
  均 forward/backward finite；LoRA 训练峰值显存约 5.84 GiB。两张 B200 很适合并发跑
  paired arm/seed，无需为该模型强行做 DDP。

### 42.4 新训练工程

- `sammed3d/peft.py`：固定六 target vocab 的 target token；12 个 image-encoder block 的
  fused QKV 仅 Q/V LoRA；adapter-aware strict checkpoint reload；组件初始化使用同一 RNG
  snapshot，保证 2x2 factorial 配对。
- `scripts/train_sammed3d_brats.py`：BF16、固定 val seed、独立 loader/sampler RNG、
  per-target val、target macro 选模、原子 checkpoint、非空输出保护、adapter metadata。
- `scripts/audit_sammed3d_checkpoint.py`：raw/adapter checkpoint 的严格 5-click clean-val
  reload，输出 row-weighted、target macro、先 timepoint 后 patient 的 subject-balanced macro，
  以及 checkpoint/manifest SHA、runtime 和 cases.csv。
- `scripts/run_worker_research.sh audit-checkpoint ...` 已接入 worker wrapper。
- `scripts/summarize_sammed3d_factorial.py` 记录 selected checkpoint SHA、factorial effect、
  seed gate 和 paired replication。
- `scripts/summarize_sammed3d_pair.py` 对 full D0/L0 做 fail-closed 汇总：强校验 protocol、
  manifest、source tree、sampler schedule、first-argmax、checkpoint payload 与 strict audit；
  audit 通过前不发布 delta，也不会声明 `pair_validated=true`。

### 42.5 2x2 subject-clean pilot 与两 seed gate

pilot 数据：
`data/datasets/brats2024_pilot_balanced_160train_fullval_patient_rr_seed20260708_v1`。
train 960 rows（每 target 160，且每 target 160 个不同 subjects；总体 387 subjects），
val 使用完整 630 rows / 67 subjects。共同设置：turbo base、3 epoch、lr/wd `1e-4`、
batch 1、5 clicks、BF16、seed2026、val seed42026、按 `val_macro_dice` 选 checkpoint。

seed2026 selected result：

| Arm | Adapter | Best epoch | 6-target macro | 4-weak macro | 相对 D0 weak | Gate |
|---|---|---:|---:|---:|---:|---|
| D0 | decoder/prompt only | 2 | 0.51938 | 0.43340 | - | control |
| D1 | + target token | 2 | 0.51781 | 0.43167 | -0.00173 | fail |
| L0 | + Q/V LoRA r4/a8 | 3 | 0.53542 | 0.45446 | +0.02107 | pass |
| L1 | target token + LoRA | 3 | 0.53855 | 0.46325 | +0.02985 | fail constraint |

- L0：3/4 弱目标不下降；SNFH `-0.00416`、WT `+0.01618`，通过首 seed gate。
- L1 虽然 weak macro 更高，但 selected epoch 3 的 SNFH 相对 D0 为 `-0.01311`，超过
  预注册最大下降 0.01；不能回头 cherry-pick epoch 2。
- 配对 seed2027 D0/L0 复现：weak macro `0.43746 -> 0.45294`，delta `+0.01548`；
  3/4 弱目标上升，SNFH `+0.01407`、WT `+0.00370`，primary 与 secondary 均通过。
- 综合结果：
  `workspace/BraTS/training_protocols/brats2024_subject_clean_factorial_results_v1.json`。
  两 seed 均为 L0-D0 正向，因此 L0 获准进入 full-scale；没有启动 GRPO。

### 42.6 strict reload 与 patient-balanced clean-val

audit root：
`/mnt/bn/liveaim/songjunyi/aflow_brats_work2_v4/clean_val_audit_seed42026`。
所有 trained checkpoint 的 reload target macro 与训练 history 精确复现（浮点误差范围内）。

| Run | Row target macro | Subject-balanced macro |
|---|---:|---:|
| raw turbo | 0.43271 | 0.42685 |
| seed2026 D0 | 0.51938 | 0.51268 |
| seed2026 D1 | 0.51781 | 0.50987 |
| seed2026 L0 | 0.53542 | 0.53113 |
| seed2026 L1 | 0.53855 | 0.52994 |
| seed2027 D0 | 0.52234 | 0.51189 |
| seed2027 L0 | 0.53562 | 0.52872 |

关键 patient-balanced delta：

- seed2026：D0-raw `+0.08583`，L0-D0 `+0.01845`，L0-raw `+0.10428`。
- seed2027：L0-D0 `+0.01682`，L0-raw `+0.10186`。

这说明 pilot 增益不是由纵向 scans 数量加权制造的；LoRA 在两个 seed 的 patient-balanced
口径也都优于 paired decoder-only。

### 42.7 full-scale exact hierarchical training

直接按 full rows 均匀抽样会让 timepoint 多的 subject 在同 target 被放大最多 7-10 倍。
因此新增 `--target-sampling target_patient_balanced`：

1. 每 epoch 六 target 各 738 次，共 4428 次；
2. target 内 patient round-robin，任意两个 patient exposure 最多差 1；
3. patient 内 timepoint seeded cycle；每 epoch 最后再做确定性全局 shuffle；
4. 每期 schedule SHA、target quota、patient exposure min/max 写入 config/history/checkpoint。

full protocol：
`workspace/BraTS/training_protocols/brats2024_fullscale_hierarchical_v1.json`。
真实 12-train/6-val GPU smoke 已通过。最终 pair 汇总与冻结前 frozen-test 安全工程全套
单测为 `181 passed`；当时真实 sealed test 尚未访问。

2026-07-17 09:27 UTC 已在两张 B200 并行启动 paired seed2026：

```text
/mnt/bn/liveaim/songjunyi/aflow_brats_work2_v4/fullscale_hierarchical_seed2026/d0_decoder
/mnt/bn/liveaim/songjunyi/aflow_brats_work2_v4/fullscale_hierarchical_seed2026/l0_qv_lora
```

共同设置：419 train subjects / 4431 rows，3 epoch，exact 4428 samples/epoch，
lr/wd `1e-4`，batch1，5 clicks，BF16，seed2026，val seed42026，完整 630-row clean val，
按 `val_macro_dice` 选 best。D0/L0 三期 schedule SHA 完全相同，仅 L0 多 Q/V LoRA。

seed2026 已完成，双方 selected epoch 均为 2：

| Arm | 6-target macro | 4-weak macro | Subject-balanced macro |
|---|---:|---:|---:|
| D0 | 0.533289 | 0.452116 | 0.527801 |
| L0 | 0.551479 | 0.468865 | 0.544192 |
| L0-D0 | **+0.018190** | **+0.016749** | **+0.016391** |

row-weighted per-target L0-D0：NETC `+0.040397`、RC `+0.018367`、ET `+0.013155`、
TC `-0.004924`、WT `+0.027598`、SNFH `+0.014544`。3/4 弱目标不下降，两个约束
目标都上升。两条 selected checkpoint 的 630-row strict reload 与训练 history 精确一致。

正式结果：
`workspace/BraTS/training_protocols/brats2024_fullscale_seed2026_pair_results_v1.json`，
SHA-256 `1ed89ea686c07421d82651767c9b6a5e5c8774ee9961abb8ed30d60adaba8418`；
`pair_validated=true`、`test_accessed=false`。

由于 full seed2026 primary 为正，seed2027 replication 已按规则晋级。启动前协议：
`workspace/BraTS/training_protocols/brats2024_fullscale_hierarchical_seed2027_v1.json`，
SHA-256 `806d77817dd52f9d2f5401d17fa1938943a707d39f19871e7187d921be3945f7`。
其 provenance attestation：
`workspace/BraTS/training_protocols/brats2024_fullscale_seed2027_prereg_provenance_attestation_v1.json`，
SHA-256 `5081c903b4ecbfbb5396acd1bfa74ff54c9222cf5612b9607e15c1309006fe2a`。该 attestation
只澄清 prereg 的 `best_sha256` 字段记录的是 `best.json` metadata SHA，不改变预注册超参、
sampler schedule、选模规则、gate、selected checkpoint 或结果。
2026-07-17 10:20 UTC 已在两张 B200 启动并完成：

```text
/mnt/bn/liveaim/songjunyi/aflow_brats_work2_v4/fullscale_hierarchical_seed2027/d0_decoder
/mnt/bn/liveaim/songjunyi/aflow_brats_work2_v4/fullscale_hierarchical_seed2027/l0_qv_lora
```

三期预注册 schedule SHA：

```text
47ea11f55d9c32e7ef4d0ba727c82475a42c96d0bdd368efe473562c2891b51d
bdb5ad02376a0361f55db33824ca61c68390c08c36d01218ae498f33a7972b22
5ad268ef907967474f11dc430cf58cc1ae88c8c45ae67972ea798c8205629770
```

seed2027 D0 选 epoch 2，L0 选 epoch 3；630-row strict reload 与 history 精确一致：

| Arm | 6-target macro | 4-weak macro | Subject-balanced macro |
|---|---:|---:|---:|
| D0 | 0.539155 | 0.465323 | 0.532617 |
| L0 | 0.560209 | 0.480259 | 0.554441 |
| L0-D0 | **+0.021054** | **+0.014936** | **+0.021824** |

row-weighted per-target L0-D0：NETC `+0.003296`、RC `+0.017289`、ET `+0.015712`、
TC `+0.023449`、WT `+0.023317`、SNFH `+0.043261`。正式结果：
`workspace/BraTS/training_protocols/brats2024_fullscale_seed2027_pair_results_v1.json`，
SHA-256 `629377f47580c772dc445f4a83a921bc097b4087ceb9d1fbcf02d10b6e65e653`；
`pair_validated=true`、`test_accessed=false`。

selected checkpoint SHA-256：D0 `0940421d36dfe29afbe2fb3974b2e836fbf52913e9ea1f122f44bb9bffecbde1`，
L0 `8c1ac50d84f4131b8c638c3b11356b6d7525ae9fd93921ad19be5f7102693e3b`。audit summary
SHA-256：D0 `c3b57885f0fc26329511a27c9eda1c176d1d8631262fb8389fa8c096aa8dc83a`，
L0 `6f1e05ec56a04b4c66035fc2dfc51724b4039478868bad651a80ef3119dc7c17`。

**接续纪律：** 两 seed full primary 和 mandatory secondary 已检查并同方向支持 L0。
frozen-test runner 的语义 cross-binding、全局一次性 ledger、TOCTOU、row identity/coverage、
implementation binding 与 COMPLETE 产物绑定均已修复并有负例；全套 `181 passed`。
GRPO 继续后置到收敛 SFT checkpoint 与 equal-step continued-SFT 对照阶段。

### 42.8 最终 L0 配置冻结

- 最终协议：
  `workspace/BraTS/training_protocols/brats2024_final_frozen_test_l0_v1.json`；SHA-256
  `9dfdf06d0ab9afbdc487f18aea677769dfe3f0d0d9abd1f0ce260eb4bb5d33f7`。
- 协议在 test 访问、global ledger 和输出目录创建前落盘；冻结时间
  `2026-07-17T11:56:33+00:00`，`test_accessed_at_freeze=false`。
- 固定五条 run：raw turbo、seed2026 D0/L0、seed2027 D0/L0；统一 5 clicks、BF16、
  audit seed42026。primary 是两 seed L0 相对 raw turbo 的 subject-balanced weak macro；
  key secondary 是两 seed paired L0-D0 的 subject-balanced weak macro。
- runner SHA-256：`25da9a3b12c6539d9b12ff14e77a27d7f76c186ec698b7e078a01305c973bb89`；
  patient-id helper SHA-256：
  `831d40e83d5f514075aa86df5d6f15fbe483e088c3d07714e0aeef565be08a66`。
- 仅非 test 的 fail-closed preflight 已验证 5 个 checkpoint、16 项 provenance 与 canonical
  split binding；此时 output/ledger 均不存在，真实 test 未读取。后续只能按协议运行一次，
  test 访问开始后 retries 固定为 0。

### 42.9 唯一 frozen-test 尝试：失败且禁止重试

- global ledger 创建：`2026-07-17T11:59:46Z`；test access 开始：`12:00:03Z`；失败：
  `12:03:12Z`，stage `checkpoint_audit:raw_turbo`。
- 根因：worker 上实际 `/tmp/aflow-sammed3d-b200-python` 只有 overlay `PYTHONPATH`，缺少
  overlay cuDNN 的 `LD_LIBRARY_PATH`。raw-turbo 推理后写 summary 时首次查询 cuDNN，发现
  torch 编译 `9.20.0`、运行时 `9.7.1`，抛出 `RuntimeError`。
- 没有 run 目录、`cases.csv`、`summary.json`、`COMPLETE.json` 或任何 test 指标。不得声称
  L0 在 test 上提升，也不得利用本次尝试做模型结论。
- 协议固定 test access 后 retries 为 0；ledger 已消费 SHA
  `55ad9ddaa03a70403026af20e88a5c32d473c105cfff4e27bc0a32dafb39f464` 对应 manifest。
  **禁止删除 ledger 或重试。** 最终外部验证需要新的、未参与训练/选择且此前未访问的 holdout。
- 证据目录：
  `/mnt/bn/liveaim/songjunyi/aflow_brats_work2_v4/final_frozen_test_eval_l0_v1_seed42026`。
  `ATTEMPT.json` SHA `ef1849aff80d9942aa6a2a02ee038df6640f9560bfccad1d5e1106c59572f628`；
  `TEST_ACCESS_STARTED.json` SHA `6c33294c3aee3c88dc497e8c1700089d98907ee086fd7ce29280c126f2ad6579`；
  `FAILED.json` SHA `7e0354d2be15ce64f0091cb5abb323b1e716244d3e3e7e731a76620275e5b720`；
  global ledger SHA `8f6dd93057a6139f2609e19326c2c8b2929d1d66683ab74b9b5b5f5ca987a96d`。
- postmortem 已把真实 CUDA/BF16/cuDNN Conv3D 初始化移到 global ledger/test access 之前；
  probe 失败不得留下 output/ledger/access receipt，冻结 summary 复用 preflight runtime。
  修补后全套 `184 passed`，runner SHA
  `9d2582c979488b4fd5787af79f088f8d659e068eb908be2477fff853bad4872d`，测试 SHA
  `1b88e491e7c38d780cca16bb9044b047825b331a919e13e4c6313c6290ef6e64`。冻结协议不修改，
  仍保留尝试时 runner provenance。

### 42.10 本地存储清理

- 已确认 `/mlx_devbox/users/songjunyi.666/playground/brats2024` 只是旧 200-scan 副本：
  200 cases / 1000 NIfTI 均在 `/mnt/bn/liveaim/songjunyi/brats2024` 的 1350 scans 中唯一存在，
  文件名与字节数全部一致。
- 该目录已删除，`playground` 从约 85G 降到 44G，释放约 41G。
- `/mnt` 全量原始数据、全量预处理、SAM-Med3D base checkpoint、venv、训练结果、协议、
  历史日志、失败回执与 test 快照均未删除。

### 42.11 后续 work #2 候选

1. 弱目标专属 decoder experts：四个弱目标确定性路由到独立 decoder；对照为同一 L0
   checkpoint 的 equal-step shared-decoder continued-SFT。
2. partial-decoder LoRA：保留当前 12-block Q/V LoRA r4/a8，只更新 decoder
   upscaler/hypernetwork output；对照为完整 decoder/prompt 的 L0。
3. 先做约 100 个六目标 macro-step 的 gradient-cosine 审计；仅在负冲突比例和 TC 冲突
   达到预注册门槛后，才做 grouped-gradient control 对 PCGrad。

统一晋级线：strict subject-balanced weak macro 至少 `+0.005`，至少 3/4 弱目标不下降，
WT/SNFH 满足非劣约束，且两个 seed 同方向。不做 QKV/rank/DoRA 网格，也不回到 work #1
的 threshold、TTA 或 postprocess 搜索。

---

## 43. 2026-07-20：work #1 审计、梯度分支与 continuation 完成

### 43.1 不可越过的边界

- 当前仍是 work #2，只允许 train/clean-val 的 SAM-Med3D 权重训练与审计。
- 不恢复 work #1 inference workflow、threshold、TTA 或 postprocess 搜索。
- 第 42.9 节的 sealed-test 唯一尝试已经消费 manifest，`retries=0`；禁止删除 ledger、
  重试、读取 test 指标或用 test 选择配置。最终外部验证只能来自新的独立 holdout。
- 历史 `label_RC` 0.58932 是旧 scan-level/subject-leaking 结果，只能追溯，不能当新 clean-val
  baseline。当前 subject-clean work #2 以第 42 节的 full-scale D0/L0 结果为起点。

### 43.2 Work #1 完成度结论

只读审计 artifact：
`workspace/BraTS/research_audits/work1_ucb_aflow_completion_audit_20260720.md`，SHA-256
`d2de9ad28fd6cd9935fc5f26c63affc109805c63b178e58cbd2ff7c37eea6828`。

结论：工程主干完成且继续冻结。实现层面的准确描述是 **UCB-guided AFlow workflow tree
search**；2026-07-20 起论文级方法名冻结为 **MedTAWS（Medical Tree-guided Adaptive
Workflow Search）**。不应直接称标准 MCTS，因为实现没有 rollout、节点 `N/W/Q` 或祖先
回传。当前缺失的是 subject-clean 外部证据和 equal-budget search-policy ablation，而不是
继续调 inference 参数。

### 43.3 Gradient-conflict 审计与唯一分支

冻结协议：
`workspace/BraTS/training_protocols/brats2024_l0_gradient_conflict_branch_v1.json`，SHA-256
`878d2cfc05de55ad5e950636c9510ab80ebc2b07a760caa97f9d961d6a96c291`。

两 seed 各在 selected L0 checkpoint 上审计同一套 100 个六目标 macro-steps；每个
target/checkpoint 100 samples，每个 group/weak-target 统计 500 个 pairwise cosines。四个
互斥原子组 `prompt_encoder/mask_transformer/mask_output/qv_lora` 完整覆盖 7,731,760 参数，
另有 `mask_decoder/all_trainable` 诊断聚合组。

输出与 SHA-256：

```text
/mnt/bn/liveaim/songjunyi/aflow_brats_work2_v4/gradient_conflict_audit_v1/seed2026_l0.json
7a42eba7b7ad04c496ac369e450c2316852f0fecb98e9be47714d9004d2712c8
/mnt/bn/liveaim/songjunyi/aflow_brats_work2_v4/gradient_conflict_audit_v1/seed2027_l0.json
9351185d8390a35a41732609d499e43999af7627631f7e76d4098ebd6b1b8ade
```

两 seed 的所有 eligible mean cosine 均为正；没有共同单元满足冻结 PCGrad gate。
`scripts/select_sammed3d_gradient_branch.py` 因此唯一输出 `partial_decoder`，common trigger=0。
决策文件：
`workspace/BraTS/training_protocols/brats2024_l0_gradient_branch_decision_v1.json`，SHA-256
`a93a98fbde1c945e623d100354b57612844d1b16c33a18b2ce61a3ad8e6d628c`。禁止 override 或
在看到 clean-val 后改回 PCGrad。

### 43.4 Continuation 实验冻结内容

协议：
`workspace/BraTS/training_protocols/brats2024_partial_decoder_continuation_v1.json`，SHA-256
`fd956ef4649422a7c2d9f01a35c8440eb66a4fd7c04e0ed44fa8fbd7a3dba3bf`。

四个 arm：

```text
seed2026_full_l0              GPU0
seed2026_partial_decoder      GPU1
seed2027_full_l0              GPU1
seed2027_partial_decoder      GPU0
```

共同设置：从各 seed selected L0 checkpoint 继续 2 epochs；AdamW state reset；lr/wd
`1e-4`；batch1；workers2；5 clicks；BF16；每期 exact 4428 train samples；完整 630-row
clean val；val seed42026；按 `val_macro_dice` 选择。candidate 可训练 793,728 参数，仅包括
12-block Q/V LoRA、`mask_decoder.output_upscaling` 和
`mask_decoder.output_hypernetworks_mlps.0`；control 保持完整 L0 7,731,760 参数。

promotion gate：两 seed subject-balanced weak macro candidate-control mean `>= +0.005`，
两 seed 各自 `>0`，至少 3/4 弱目标两 seed均值不下降，WT/SNFH 满足每 seed 与均值非劣约束，
六目标 subject-balanced macro 均值不下降。失败即记录 negative result；通过也只晋级为
clean-val candidate，不授权 test。

### 43.5 实现与验证绑定

- 历史 `scripts/train_sammed3d_brats.py` 必须保持原字节，SHA-256
  `0c5026e98b2e0f848c16b436610e9b7eca391995c062ac128ce1bee26d1bfa4f`。
- 新 runner `scripts/train_sammed3d_continuation.py` SHA-256
  `993421f36ee5d58b8269783437d65c292e390ecba1e373056d0f3de8619a435b`。
- runner tests `tests/test_train_sammed3d_continuation.py` SHA-256
  `7abdfe3f2838c3ac9bc9f3f7e511c1174d1ac07df266caee8634d2cb460643e5`。
- exact launcher `scripts/launch_sammed3d_continuation_v1.sh` SHA-256
  `45e1ac5c940c0ad46cf0f16b31d328ac9f6a6f11e716a61b8d3c0ad2602297f9`。
- 初始完整 CPU test suite：`205 passed`。B200 两卡 preflight：torch 2.12.0+cu130、CUDA13、
  SM100、BF16、cuDNN92000、Conv3D forward/backward finite。

### 43.6 基础设施失败与唯一恢复

第一次提交于北京时间 `2026-07-20 12:05:45`。非交互 SSH 环境的
`LD_LIBRARY_PATH` 为空，动态链接器选中了
`/usr/local/cuda-12.6/compat/libcuda.so.560.35.05`，而不是系统 580.105.08 driver library。
因此四个进程都在 runtime preflight 报 `continuation protocol requires CUDA`；train/val
manifest、模型和 optimizer 均未加载，训练步数为 0，arm 输出目录也未创建。原始 receipt
与 logs 均保留。

冻结失败证明：
`workspace/BraTS/training_protocols/brats2024_partial_decoder_continuation_attempt1_failure_v1.json`，
SHA-256 `7786b321a481b793adb32725c12ec61726b4ddcdd3dec74bb53c5551414284ee`。

`scripts/resume_sammed3d_continuation_v1.sh` 将
`/usr/lib/x86_64-linux-gnu` 前置，并强制两张 B200 都通过 CUDA、SM100、BF16、cuDNN92000 和
finite Conv3D forward/backward preflight；SHA-256
`b4e1cc65e3a75ac079b9d095dd099fff9e8a5dad42ffa17dcca1a502317c3296`。科学协议与 runner
未改变。唯一一次 infrastructure-only resume 于北京时间 `12:12:25` 启动，attempt2 receipt
SHA-256 `dccd7ee5d5509d41fd07d809178f486cc31e1034256ca6db134679bc09ed557d`。

四臂均跑满 2 epochs，并按冻结 earliest-argmax 规则选择 epoch 1。完成时间依次为：
seed2027 partial `12:41:52`、seed2026 partial `12:42:39`、seed2027 full `12:46:06`、
seed2026 full `12:46:23`（均为北京时间）。selected checkpoint SHA-256：

```text
seed2026_full_l0          8dc2ca64574cd4679839029ced707dec2ae2b74775248956ec300546f887120b
seed2026_partial_decoder  8690dd822cc60f5ebcd6299cb35750b10f1cbf56fad8253baefd2a0834d549d4
seed2027_full_l0          de5f9a96b6d273c55e4f17edd815ca1028ed304b8fc82380a66b9c9f779c4b23
seed2027_partial_decoder  09b210cd163c6602143552789226a9260fdda2edec93859c5ca907ecbd18e092
```

### 43.7 独立 strict clean-val 审计与结论

审计协议在训练完成前冻结：
`workspace/BraTS/training_protocols/brats2024_partial_decoder_continuation_audit_v1.json`，
SHA-256 `72062112ba907d86b7ae45cb20a0ab93883a5740d3afe2beb08700d408d3f10e`。
独立汇总器 `scripts/summarize_sammed3d_continuation.py` SHA-256
`d6515fd41f93ec550534e0ed5baf8c3a708c2944763d746f8244b9edaf2d9f4e`。四个 selected
checkpoint 于北京时间 `12:46:59` 后分别重新加载，在 val seed42026 上完整评估 630 rows、
133 scans、67 subjects；没有读取 test。

| Seed | Control weak | Partial weak | Delta weak | Delta six | Delta row target macro |
|---:|---:|---:|---:|---:|---:|
| 2026 | 0.466649 | 0.472120 | +0.005471 | +0.006798 | +0.010186 |
| 2027 | 0.468040 | 0.473098 | +0.005058 | +0.005664 | +0.006833 |
| 两 seed 均值 | - | - | **+0.005265** | **+0.006231** | **+0.008510** |

六目标两 seed subject-balanced 平均 delta：NETC `+0.003185`、RC `+0.001140`、ET
`+0.014536`、TC `+0.002197`、WT `+0.005915`、SNFH `+0.010413`。所有冻结 gate 条件
通过，最终 `decision=promote_clean_val_candidate`。但 weak macro 仅比 `+0.005` 门槛高
`0.000265`，且 seed2026 的 RC/TC、seed2027 的 NETC 单独看为负，故不得夸大为逐目标稳定
改善或外部有效性结论。

冻结结果：
`workspace/BraTS/training_protocols/brats2024_partial_decoder_continuation_results_v1.json`，
SHA-256 `3daefb85a8bb588cbf63470171cd59251f701a84534e12925a04249a6db8abee`。
结果中 `authorizes_test_access=false`；通过 gate 仅授权把 partial decoder 保留为下一阶段
clean-val candidate。新增汇总测试后完整 suite 为 `211 passed`。

### 43.8 当前运行状态与接续约束

四个训练进程和四个审计进程均已正常退出；无 SSH、worker-login、训练或审计后台终端。
旧 Codex 与三个子 agent 均已结束，只保留当前 Codex 和维持 2xB200 配额所需的
`mlxc worker launch`。不要重跑 continuation、audit 或汇总器，也不要删除 attempt1 证据。

下一步仍属于 work #2：冻结本轮 candidate 和全部 provenance，优先做不接触 sealed test 的
独立复核设计。由于晋级余量很小且存在单 seed 目标符号反转，在获得新的 subject-disjoint
外部 holdout 或预注册 confirmatory train/clean-val replication 前，不应解封 test，也不应
回到 threshold、TTA、postprocess 或 inference workflow 搜索。

---

## 44. 2026-07-20：MedTAWS 更名、1350-scan 文献对标与科研目标重置

### 44.1 Work #1 名称冻结

Work #1 的论文/答辩方法名改为 **MedTAWS（Medical Tree-guided Adaptive Workflow Search，
医学树引导自适应工作流搜索）**。建议论文题目：

> MedTAWS: Tree-guided Adaptive Workflow Search for Automatic Multi-modal 3D Medical Image Segmentation

这次只改展示层名称，不改算法。上面的 automatic multi-modal 标题覆盖早先 naming artifact 中
基于旧 interactive backend 的 `Promptable` 副标题；代码、repo、日志和历史 artifact 中的
`AFlow` 全部保留。
首次论文定义必须说明 MedTAWS built on AFlow；准确算法仍是 UCB-guided workflow tree
search，不是标准 MCTS。

冻结命名 artifact：
`workspace/BraTS/research_audits/work1_medtaws_naming_decision_20260720.md`，SHA-256
`083fa3de5fdeaafc40b33069cad15b7115c835ed8bd71e2b5e8af9f704173b35`。

### 44.2 外部 benchmark 的确定结论

当前 partial-decoder 两 seed strict clean-val subject-balanced 值为：ET `0.483358`、NETC
`0.301040`、RC `0.575759`、SNFH `0.679483`、TC `0.530278`、legacy WT `0.757572`，
六区 macro `0.554582`。

官方 2024 冠军 hidden-test lesion-wise Dice 为：ET `0.7900`、NETC `0.8076`、RC
`0.7760`、SNFH `0.8926`、TC `0.7874`、WT `0.8938`，descriptive macro `0.82457`。
当前 raw gap 六项均为负，macro 约 `-0.2700`。**目前没有超过官方冠军或已调研的公开
1350-scan 工作。**

但这些数字也不具备正式可比性：当前是 GT-error corrective 5-click、单模态、逐 binary
target、foreground-positive-only、standard voxel Dice，且 legacy WT `{1,2,3,4}` 含 RC；
官方是四模态、无提示、全病例、多类输出、lesion-wise Dice/HD95，WT `{1,2,3}` 不含 RC。
历史 `label_RC=0.58932` 还来自旧 subject-leaking split，严禁进入外部对比表。

完整调研和来源：
`workspace/BraTS/research_audits/brats2024_gli_1350_benchmark_review_20260720.md`，SHA-256
`65242dc51894d909096ff06ca2240247faff51973f5a2d7fbf21bce19ba24397`。

### 44.3 Work #2 主问题重置

从本节起，Work #2 headline 问题冻结为：

> 四模态、全自动、多类 SAM-Med3D 能否在 BraTS 2024 post-treatment GLI 上，以官方区域和
> lesion-wise 指标超过同协议 nnU-Net/MedNeXt 强基线，并在一次性外部 holdout 上确认？

原有 5-click binary 权重训练保留为 interactive prompted delineation 辅助实验，不再以其
clean-val Dice 判断是否超过自动分割论文。后续任何训练必须先服务于 official automatic
track，而不是继续追逐原协议的小数点提升。

### 44.4 下一阶段固定顺序

1. 数据 provenance：核实 Kaggle 三分包是否对应官方 2024-06-17 修正版 v2；官方 v2 zip
   MD5 为 `1d910b17d6cd32e38aa6296b8dfb7c77`，v1 有 10-15 个错误标签。
2. 官方数据工程：四模态、1 mm、613-subject grouped folds、全病例、四原子类，派生 TC 和
   不含 RC 的 WT。
3. 官方评测工程：接入并单测 BraTS 2024 lesion-wise Dice/HD95；保留 subject-balanced
   secondary summaries。
4. 强基线：在相同 folds 上训练 nnU-Net v2/ResEnc，资源允许再加 MedNeXt。
5. SAM 主方法：四模态 fusion + joint semantic head 或完全自动 proposal/prompt；evaluation
   不得读取 GT 生成 click。
6. 外部确认：只有方法、checkpoint、metrics 和统计完全冻结后，才核实并考虑 Synapse
   additional 271。它目前只能称 candidate post-challenge holdout，不能未经证明称原 hidden
   test。

两张 B200 适合把 paired folds 或 baseline/method arms 并行跑，但在第 1-3 步完成前不要盲目
启动长训练。当前 BraTS 2024 submission queue 已关闭；仅在 1350 内部切 val 不能声称超过
原 leaderboard 冠军。

### 44.5 延续禁令与运行状态

- 不回 work #1 threshold、TTA、morphology、postprocess 或旧 inference workflow 搜索。
- 不访问、重试或重命名复用当前 consumed sealed test。
- 不把 5-click/positive-only/legacy-WT 数字写成官方 leaderboard 指标。
- 不在 external holdout 访问后继续选架构或调参。
- 本轮只做文献审计和文档落档，没有启动训练、推理、SSH 或 GPU background terminal。

---

## 45. 2026-07-20：四模态统一重启 Phase A 完成

### 45.1 Work #1 复核后的唯一结论

用户即时纠正：从 Work #1 开始，医学任务输入必须始终是按固定顺序
`(t1n, t1c, t2w, t2f)` 联合输入。逐层复核确认，历史 backend 是单模态、GT-centered crop、
GT-error corrective click、逐 binary target 和 positive-only standard Dice，不能继续承担四模态
自动多类论文证据。因此当前准确表述是：

> MedTAWS 搜索算法原型及历史闭环保留；BraTS 医学任务接入整体作废并按四模态自动多类协议重建。

MedTAWS 仍是 stochastic UCB-guided workflow-tree search，不是标准 MCTS。新契约冻结 fold、
prompt、objective、manifest、checkpoint、code、operator/LLM 和 RNG provenance；递归拒绝
`seg/label/gt/groundTruth` 及其嵌套别名，并用四臂 lockstep write-ahead meter 做严格等预算消融。
审计 artifact：
`workspace/BraTS/research_audits/work1_four_modal_restart_audit_20260720.md`，SHA-256
`c6a2618a74cd15853618eb7868c661bd4e15048c006ae96008f4d4360aafc91e`。

### 45.2 数据事实、sealed 边界与 provisional 协议

`/mnt/bn/liveaim/songjunyi/brats2024` 的聚合体检得到 1350 scans / 613 subjects / 6750 files，
每例严格包含四模态和 seg；全部为 `182x218x182`、1 mm、LAS，同例 affine/shape 对齐且标签域
为 `0..4`。该源级体检读取了所有原始文件，因此物理上也覆盖历史 269 例；但没有打开 sealed
manifest/ledger、识别这些病例、运行预测或计算指标。它们仍不得重新包装为 clean evidence，
additional 271 仍未访问。

当前仅生成 provisional 开发协议：948 train scans / 419 subjects 和 133 val scans / 67 subjects，
case/subject 均无交叉；总计 1081 scans / 486 subjects。不要把它描述成最终 1350 全量协议，也
不要用全源标签频率选择 loss。manifest：
`data/datasets/brats2024_four_modal_development_provisional_v2/manifest.json`，SHA-256
`840c27488c1650a7c1b66a841b2e3b0a30d8780afae11112bbd4ded442f0fc52`。

本地 Kaggle `.complete` 仅证明 Kaggle dataset version 2，不足以证明 Synapse 2024-06-17 修正
v2。官方修正 zip MD5 固定为 `1d910b17d6cd32e38aa6296b8dfb7c77`。新 verifier fail closed，
但在取得官方 zip 且明确批准完整 source hashing 的 sealed-boundary 影响前，不要对真实 1350 源运行。

### 45.3 新四模态工程入口

| 组件 | SHA-256 |
|---|---|
| `sammed3d/brats_data.py` | `00613019723cb88a7bdc5566fb8504cac1d89fd8701a7d0d4d2f67e05fcc5759` |
| `sammed3d/brats2024_fourmodal.py` | `7e6aa267da1486087f754cdd569d0aef716a64c213082b1c066039f262cbe13d` |
| `sammed3d/brats_metrics.py` | `dc3c02c35b1deb22d687cc1c5fd76565aed264b300cf4aa4aca8c87320b74ded` |
| `sammed3d/four_modal.py` | `2c203edff127a65e88e94b50fd62fb1d2ffc183d3fcabfb0297422c800c5b4cd` |
| `sammed3d/four_modal_training.py` | `b6052c710e1fab6097b91ce776933a6f03e3f5b8029c8805f4acea75148e9a39` |
| `medtaws/search.py` | `119c90367dafff161fca37255e3cbe1b3bfddc7ee8bbb0b503608c8f7475b716` |
| `medtaws/ablation.py` | `11560fcf11b5a41339f03b6d71d6708bd20d8bf150bfe4339a86e410f41167d1` |
| `scripts/build_brats2024_four_modal_manifest.py` | `31313f8436b5629f7cd7c5b0d6e0ab988f4a7b1b69cc08b7ae8ab50f155a8a68` |
| `scripts/verify_brats2024_source_provenance.py` | `fa24fe037a3153083220001ca7ec6bb6f984a86aad5528eabc16eeacf5c0f8fa` |
| `scripts/smoke_sammed3d_four_modal.py` | `f9a603364191b615fc8cac08c61c8955c9e75377a286bdcd6b046980c8f40646` |

loader 将 inference case 与 supervised case 分型；validation crop 只依赖 image/brain mask，
label-balanced patch 仅允许用于 train，并记录 full-grid restore metadata。M1 严格载入历史 1-channel
checkpoint 后用 `W/4` inflation 改为四通道；C0 是等权初始化的 learned scalar gate。两者共用
无 point/box/mask prompt 的四 mask-token 原子语义头：NETC/SNFH/ET/RC，背景 logit 固定为零，
派生 TC `{1,3}` 和官方 WT `{1,2,3}`。固定 loss 为 `CE 1 + atomic Dice 1 + TC 0.25 + WT 0.25`。

`brats_metrics.py` 实现 BraTS 2024 公共算法所需六区、3-voxel dilation、50 mm3 GT threshold、
374 HD penalty 和 26-connectivity；在拿公共仓库真实 case 做逐项 parity 前，只能称
public-algorithm implementation，不能称 official bit-exact。

### 45.4 B200 验证与运行状态

两张 NVIDIA B200 上已并行完成真实 SAM-Med3D base checkpoint、BF16、输入
`[1,4,128,128,128]` 的 forward/loss/backward。最终 canonical v2 报告均生成于北京时间
`2026-07-20 15:42:48`：

| Adapter | Loss | Peak CUDA memory | 报告 SHA-256 |
|---|---:|---:|---|
| C0 | 2.953255 | 6,319,043,584 bytes | `46f036363af0c6a69e12edca50d336e904ac04ca32a5780ee1169c3b800d36cb` |
| M1 | 2.962843 | 6,388,141,056 bytes | `93686e7fcfcb7848eca155e9f1d701cac1d108b7fb0131998ab4d3ad94ba100a` |

报告位于 `workspace/BraTS/training_protocols/brats2024_four_modal_b200_smoke_{c0,m1}_v2.json`。
C0 gate、patch embed 和 decoder 梯度有限且非零；M1 patch embed 和 decoder 同样通过。完整测试
`320 passed`，新增模块 `py_compile` 和 scoped `git diff --check` 通过。无 manifest builder、训练、
评估或 SSH 后台进程；只保留维持 2xB200 allocation 的 worker launcher。v1 smoke 报告保留为
早期成功证据，但已由 v2 supersede，不得当 canonical report。

### 45.5 下一次接续顺序与禁令

Phase A 协议：
`workspace/BraTS/training_protocols/brats2024_four_modal_restart_phase_a_v1.md`，SHA-256
`b4e49b957391df4e4ebf522e188c43e105795666c808f7c0e4c5268100254f48`；当前
`authorizes_formal_training=false`、`authorizes_test_access=false`。

下一次严格按以下顺序继续：

1. 取得官方修正 v2 source parity 证据，或明确记录无法取得时的可发表 provenance 限制。
2. 用 BraTS 公共评测仓库真实 case 完成 six-region Dice/HD95 executable parity。
3. 冻结 provisional manifest 后做 protocol-bound 2-case real-data trainer smoke，不先跑正式长训练。
4. smoke 通过后预注册 paired C0/M1 pilot，并在相同 folds 上启动 nnU-Net/ResEnc 强基线。
5. checkpoint 和 metric 全部冻结后，才允许 MedTAWS 自动多类搜索及四臂等预算消融。
6. additional 271 只允许在最终配置先冻结后一次性确认；当前 consumed sealed test 永不重试。

继续禁止 threshold、TTA、morphology、postprocess 和旧 inference workflow 搜索；不得用全 1350
标签频率调 loss，不得把 provisional 1081 写成全量训练，不得把本轮 B200 smoke 写成新 Dice 或
正式训练结果。

---

## 46. 2026-07-21：Phase B 指标 parity、真实 smoke 与 paired split

### 46.1 lesion-wise scorer 状态

公开参考仓库 `https://github.com/rachitsaluja/brats_val_2023` 已冻结到 commit
`43c905242b2eecf421d4ab2da7af8ece9777d322`。parity runner 在隔离进程使用参考仓库自带的
surface-distance；只用 `np.Inf=np.inf` 兼容 NumPy 2，没有修改参考文件。13 个合成对抗 3D cases
的 Dice、HD95、TP/FN/FP/evaluated-GT 最大绝对差均为 0。

报告：`workspace/BraTS/training_protocols/brats2024_metric_parity_v1.json`，SHA-256
`8c1754ea3ff36df0529e45f64dc6958a7cc4056d489827e732009cff1b31a1e8`；runner SHA-256
`130fd1233f626f2b5318cbd7a6f4f230856e738509d716da7b3d3506a38ae791`。仓库没有 organizer
real-case fixture，因此准确表述是“与公开 scorer 在冻结测试矩阵 executable parity”，不是
不可见 leaderboard backend 全环境 bit identity。

### 46.2 source provenance 仍未解锁

官方 corrected-v2 实体：`syn60086071` version 2，file handle `143397620`，modified
`2024-06-17T21:24:56.161Z`，zip MD5 `1d910b17d6cd32e38aa6296b8dfb7c77`。本机无官方 zip 和
Synapse credential。三份本地源是 2026-05-28 上传的 Kaggle version 2，每份 note 为 450
patients；三个 `bundle.complete` 都是零字节，不能证明与官方 zip 同一。

机器可读状态：`workspace/BraTS/training_protocols/brats2024_source_provenance_status_v1.json`，
SHA-256 `d85e2d3f6a2acceeda1e5f015b6c6d6a13bf74d72083930093e42db0b962f47c`。在取得 official zip、
核对 MD5 并执行 archive/local per-file SHA-256 parity 前，`authorizes_formal_training=false`。

### 46.3 真实数据 train-save-reload-full-grid smoke

固定 cases：`BraTS-GLI-00005-100` 与 `BraTS-GLI-00008-100`，来自不同 subject；manifest SHA-256
`965adc6f9b02f6d453141cd14612eb703edacdcf68204a5c43c0986a5de4ca4d`。相同 seed `20260721`、
128 cube、AdamW `1e-4`、weight decay `1e-2`，每臂两个 optimizer steps。

最初 CUDA memory reset preflight 在任何数据/模型前失败；第二次误用 SM90-only base venv，C0
到首次 forward、M1 到 patch inflation 失败，均零 optimizer step、无 artifact。确认
`/tmp/aflow-sammed3d-b200-python` 为 Torch 2.12+cu130、cuDNN 92000、SM100 且双卡 Conv3D
通过后，仅重提一次并成功：

| Arm | 两步 loss | delta norm | peak CUDA | report SHA-256 | checkpoint SHA-256 |
|---|---|---:|---:|---|---|
| C0 | 2.903346, 2.875066 | 2.786844 | 6,467,155,456 | `0580f53ddfaa2bdd313b453979248f062b41cb749f9c8cf48eb2c2ac591e67ee` | `aefb194e811af1ce635f0ac43ea13f7f56f2e7111f4d12048ed1cfa2f87f27fd` |
| M1 | 2.903374, 2.875078 | 3.080492 | 6,651,086,336 | `3ad4e5126eb4f49e11e95fb321e229800d239988d74bcc95f1bee2134548798d` | `a5a39d06f325b87f4fd7ccd8614a99cbb1b82d00db79b8ad51ac3642ba959206` |

两臂 checkpoint strict reload，label-free inference 对 133x162x132 brain crop 做 8 windows 后恢复
182x218x182。两步后输出都为全背景；只能证明工程通路，不能晋级模型。报告固定
`metrics_computed=[]`、`authorizes_formal_training=false`、`authorizes_test_access=false`。

新增入口：`sammed3d/four_modal_inference.py` SHA-256
`a704a78068a835e9a1a759681f485f03d1bd20586809140ac0d9fbba1ee53e96`；training SHA-256
`b91a6c850f25c5b5c0775b14fe46bf8abbb6576f7a05948c804b8d202542ebb2`；real smoke runner SHA-256
`93a2b5f7421b86cc0ecac95f12738a7d86858983bb9a27c8c7a3ff1a2108173e`。

### 46.4 paired pilot split

为阻止 N0 fingerprint/selection 读取 confirmatory val label，948-case development-train 已按
subject SHA-256 ranked split：

| Split | Cases | Subjects | 用途 |
|---|---:|---:|---|
| fit | 764 | 335 | fingerprint、参数优化 |
| tune | 184 | 84 | checkpoint 选择、pilot 决策 |
| confirmatory val | 133 | 67 | pilot 冻结后一次性开发集确认 |

subject overlap 全为 0。manifest：`data/datasets/brats2024_four_modal_paired_pilot_v1/manifest.json`，
SHA-256 `eaed5710dce8f10f6367b531d4161323028a7c6f449f3e702f7b7cb4b0e2aa35`。builder SHA-256
`687d261cb6eccf2fd80c88155d3ddb24c3cb211df590adb5d2d93b506a6fd1bf`。

### 46.5 当前状态与下一步

完整 suite `327 passed`；worker 无训练、评估或 GPU compute process，仅 allocation launcher 保留。
完整审计：`workspace/BraTS/research_audits/brats2024_phase_b_readiness_20260721.md`，SHA-256
`5bf1a1d06d9e75c07b0f10a40a9657d7f021fd36f315e31971ed01829bc7ea0c`。

下一步允许做 fit-only nnU-Net v2 ResEnc N0 dataset/fingerprint 工程和 preregistration，但不得启动
正式 C0/M1/N0 长训练，直到 official archive parity 通过。133-case confirmatory val 在 pilot 配置
冻结前不得用于 selection；test/additional 271 继续封存；MedTAWS 等待自动 checkpoint 冻结，且
继续禁止 threshold/TTA/morphology/postprocess 搜索。

---

## 47. 2026-07-21：Work #1 冻结四模态 base 先测完成 tier1

### 47.1 本节覆盖的旧顺序

用户明确纠正：先回答冻结 base 经 MedTAWS 后是否提升，再启动 Work #2 训练。因此第 46.5 节
“MedTAWS 等训练 checkpoint”的口径失效。source checkpoint 保持冻结，C0/M1 构造不属于训练。
当前协议：
`workspace/BraTS/training_protocols/brats2024_work1_frozen_fourmodal_medtaws_v1.md`，SHA-256
`c3aa4fe7bdfbf0ec10a3708776a248dda5c02314ce212bbe5f9afd27c66973b4`。

协议只授权 184-case tune baseline/search；133-case confirmatory、test、additional 271 和权重
训练均未授权。用户要求不再继续数据来源调研；开发实验按现有四模态数据相同处理，既有 archive
parity 限制只保留为论文 provenance 说明，不再作为本开发链路的执行阻塞项。

### 47.2 冻结 baseline 数值

source checkpoint SHA-256：
`899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`。双 B200 于北京时间
2026-07-21 14:16 至 14:38 并行评测 184 scans / 84 subjects，0 optimizer、0 gradient updates，
全部 state_dict 参数与 buffer 前后 exact-byte hash 一致。

| Arm | 六区 case macro Dice | HD95 | subject macro Dice |
|---|---:|---:|---:|
| C0 | 0.082263 | 262.4058 | 0.081746 |
| M1 | **0.083186** | **261.8209** | **0.082768** |

M1 逐区 Dice：ET `0.000026`、NETC `0.385870`、RC `0.016339`、SNFH `0.044452`、TC
`0.000026`、WT `0.052404`。NETC absent-case credit 抬高总 macro；冻结无 prompt base 的实际
肿瘤分割能力很低。

reports：

- `workspace/BraTS/frozen_baseline/c0_tune184_v1.json`，SHA-256
  `6e1cd67d4848bc500dd01540e872e3651bc62bf581e34347c8ede5ecf8be9cd6`
- `workspace/BraTS/frozen_baseline/m1_tune184_v1.json`，SHA-256
  `ada9907883b17ab7b7c6f4e8c8d0f08966460751d1570b0b861dc623012ce36a`
- baseline summary，SHA-256
  `405b1c26e35e44cf1a8a1cffe0e1cf7ab7b783eb4569d60de6636202333d0d2b`

### 47.3 MedTAWS tier1 结果

canonical raw-token caches：C0 SHA-256
`d5cc69f34c84990db2e6b40aa7cbd7bcbdd9f40e7c544ad8f50b8b1a8ee939fe`、M1 SHA-256
`0625e36e7f4c0d1520b36f68cd5075a7bb491c3239c83e5977aa7cc10698409a`；各 184 cases / 1456
atomic window forwards。model 进程只读 label-free workflow manifest；GT 只在外部 scorer 中读取。

tier1 catalog 是 native C0/M1 × 24 个全局 token mapping，共 48 workflows。五个固定 seeds
`20260721..20260725`，每 policy/seed 12 draws、12 cache reads、2208 logical backend units；
search physical model forwards=0、LLM calls=0，四臂预算严格相同。

| Policy | final-best mean | 相对 frozen M1 | normalized AUC | oracle hits |
|---|---:|---:|---:|---:|
| MedTAWS | 0.092935 | +0.009749 | 0.090859 | 4/5 |
| score-softmax | 0.092754 | +0.009568 | 0.091287 | 3/5 |
| greedy | **0.093116** | **+0.009930** | **0.091875** | **5/5** |
| uniform | 0.092573 | +0.009387 | 0.091084 | 2/5 |

直接回答：冻结 base 经 MedTAWS 后在 tune 上**有提升**，但 MedTAWS 没有胜过等预算 greedy。
post-search oracle 是 `v1.m1-native.p-nesr`，mapping 为 raw tokens →
`NETC,ET,SNFH,RC`，Dice `0.093116`。分解为 structure gain `0`、token mapping gain
`+0.009930`、joint `+0.009930`、interaction `0`。因此本轮增益全来自全局语义排列，不能写成
multi-scale/coarse-refine 或树策略的结构贡献；绝对 Dice 仍远低于外部目标。

完整 report：
`workspace/BraTS/medtaws/work1_cached_tier1_search_tune184_5seed12_v1.json`，SHA-256
`f651d67f83261797c8e6e35f22b3c29211f56b091bda86d83e07007373fb7aa0`；compact summary SHA-256
`0fd00297292d8d983f88bd71175f4f4b50f538908ba51711de5db6e2d461714b`。

### 47.4 算法/工程修正

旧 equal-budget controller 使用不可变候选快照，新增 node 不会进入下一轮，不能称动态树。本轮
已修为每 policy 独立 tree pool：receipt 原子加入 child，selected parent 的 improving/
non-improving child 和 visit proxy 更新，新 score 进入下一轮 UCB/softmax/greedy，candidate
digest 与 tree-update audit 逐轮记录。有限 catalog 对每 policy 去重，邻域耗尽后确定性 fallback，
全层耗尽 fail closed。

operator set SHA-256：
`12b6b4e0f8976091ed9d52d6cb7695af6bbb577a3128cc9d3926db3f6d4c472d`。完整 suite
`400 passed`；warnings 仅为 `surface-distance` 的 SciPy deprecated namespace。

另已实现并 synthetic 验收 7 个真实 cost-tier2、2-unit 模板：C0/M1 multiscale、cross-model
uniform fusion、C0/M1 coarse-to-refine、cross-model/cross-scale uncertainty routing。它们只完成
工程，不曾运行真数据，不得写成性能结果。

### 47.5 下一次唯一顺序

1. 为 tier2 的少量共享 primitive tasks 构建 raw-logit cache，避免按 168 workflows 重复模型推理。
2. 在 184-case tune 上按同 cost tier 运行五 seed 四臂，分解结构、token mapping 和 interaction。
3. 只有 tier2 workflow、seeds、代码和统计冻结后，才生成 confirmatory unlock artifact，并一次性
   比较 frozen M1、tier1 control 和最终 MedTAWS；在此之前不得评分 133-case confirmatory。
4. Work #1 结果/失败分析完整落档后，另立 Work #2 四模态权重训练协议。

继续禁止 threshold、TTA、morphology、connected-component 和 postprocess 搜索；test/additional
271 继续封存。当前无 GPU compute、搜索、训练或多余 background terminal，只保留 2xB200
worker allocation。

---

## 48. 2026-07-21 17:03:06（北京时间）：A1 baseline v2 单 seed preliminary

> **历史 preliminary，已由第 49 节五 seed final supersede。** 本节保留作 provenance；
> 48.4 中“继续跑剩余四 seeds”已经完成，不再是下一步。

### 48.1 必须覆盖第 47 节旧执行建议的纠错

第 47 节的 no-prompt token permutation baseline/MedTAWS 已因 mask-token semantic contract 无效而
撤销；随后建立的 oracle-interactive v1 whole-brain smoke 也已撤销。后者失败根因冻结为：

1. 四分支 fused decoder lowres 先上采样 `128^3`、再下采样反馈，未 exact carry decoder lowres；
2. whole-brain bbox resize 改变 checkpoint 所见 native-spacing target-centered crop 的尺度与 context；
3. zero-preserving population-std normalization 与 upstream TorchIO fixed-window、`x>0` statistics、
   unbiased sample std、full-tensor transform 不一致。

旧失败 artifact：
`workspace/BraTS/prompt_binary_native4_late/smoke_seed20260721_case1.json`。只作工程失败记录，
不得恢复执行或引用为性能证据。

### 48.2 当前唯一授权 baseline 协议

v2 使用 `a1_first_ritm_click`：首个 full-grid RITM 正点击成为 native-spacing `128^3` 固定窗中心；
click 2 在窗内用 RITM，click 3-5 在窗内 uniform error；四模态各自执行 upstream-exact
normalization；四个冻结原生一通道分支等权融合 decoder low-resolution logits，并将 exact fused
lowres 原样用于下一轮 dense mask prompt。预测恢复到 full grid，窗外 target 全部计 FN。

协议：`workspace/BraTS/training_protocols/brats2024_sammed3d_oracle_interactive_baseline_v2.md`；
文件 SHA-256 `408bb8dfd185dcc296f113f46b7315f6c6d552fd5112c948ac7e02294b2a1f15`。

### 48.3 已完成的 seed 20260721

严格完成 development tune 184 cases / 84 subjects；present-query denominators 为 ET 129、NETC 72、
RC 147、SNFH 184、TC 130、WT 184，共 846 queries。

| Metric | @1 | @3 | @5 |
|---|---:|---:|---:|
| D6 case macro | 0.282351 | 0.386175 | 0.418646 |
| D3 ET/TC/WT | 0.289645 | 0.393663 | 0.431232 |

区域 Dice@5：ET `0.361107`、NETC `0.226620`、RC `0.550598`、SNFH `0.440961`、
TC `0.429663`、WT `0.502927`。coverage mean/min=`0.986089/0.501321`；`148/846` queries
没有覆盖完整 target，但窗外 FN 已进入 full-grid Dice。

merged report：
`workspace/BraTS/prompt_binary_native4_late/a1_torchio_v2_seed20260721_tune184_merged.json`，
SHA-256 `78998bfd21c1afff0a5aa550271bc6591ddd4421e8be80759180cf4f7e21c405`；内嵌 canonical
protocol SHA-256 `d4d34b6a1e876b3742616d3c752708d28de39fd399c9c0f44ed1b562969b0673`。

### 48.4 当前硬边界与下一步

这是固定 5 seeds 的 `stage 1/5`，只可称 **single-seed preliminary development-tune result**。
下一步只允许按同一 v2 协议完成 `20260722,20260723,20260724,20260725`，然后汇总五 seed mean
与 population std。四个 seed 完成前不得选择 seed、改协议或把 D6@5 `0.418646` 当最终值。

本协议 `authorizes_medtaws_search=false`、`authorizes_weight_training=false`、
`authorizes_confirmatory_val=false`、`authorizes_test_access=false`。因此 MedTAWS、训练、
confirmatory、sealed test 和 additional 271 都不得启动；当前记录没有解封其中任何一项。

---

## 49. 2026-07-21 17:39:58（北京时间）：A1 baseline v2 五 seed final

### 49.1 完成状态

冻结协议的五个 seeds `20260721,20260722,20260723,20260724,20260725` 已全部完成，状态从
`stage 1/5 preliminary` 更新为 **baseline complete 5/5**。每个 seed 严格运行相同 184 cases /
84 subjects，聚合使用 raw-case recompute、五 seed arithmetic mean 和 population std。

| Metric | @1 mean ± pop std | @3 mean ± pop std | @5 mean ± pop std |
|---|---:|---:|---:|
| D6 case macro | 0.281557 ± 0.004033 | 0.382280 ± 0.004517 | 0.416978 ± 0.004925 |
| D3 ET/TC/WT | 0.289178 ± 0.006505 | 0.389753 ± 0.006895 | 0.430292 ± 0.008256 |

region Dice@5 mean ± population std：ET `0.367905±0.006310`、NETC
`0.226707±0.007179`、RC `0.534767±0.010500`、SNFH `0.449514±0.006454`、TC
`0.436665±0.004577`、WT `0.486307±0.020216`。

| Seed | D6@5 |
|---:|---:|
| 20260721 | 0.418646 |
| 20260722 | 0.415488 |
| 20260723 | 0.422110 |
| 20260724 | 0.408165 |
| 20260725 | 0.420479 |

### 49.2 canonical artifact

aggregate path：
`workspace/BraTS/prompt_binary_native4_late/five_seed/a1_five_seed_tune184_aggregate.json`  
SHA-256：`145067a8e77594d9c0c6e9274d6158774aab95bc42196c4ec8afce579acf1438`  
frozen protocol document SHA-256：
`408bb8dfd185dcc296f113f46b7315f6c6d552fd5112c948ac7e02294b2a1f15`

### 49.3 解释边界与后续权限

本结果的任务定义仍是 `oracle-interactive`：六个 region 分别作为 binary query，GT 提供首次正点
及后续 corrective clicks，只在 region-present cases 上计算 full-grid voxel Dice。它不是 automatic
BraTS multi-class segmentation，不能与 automatic BraTS leaderboard 或论文指标直接横向比较。

完成 `5/5` 只代表冻结 baseline 的 development-tune 重复完成，不自动授权下一研究阶段。artifact
明确 `automatic=false`、`medtaws_search=false`、`weight_training=false`、
`confirmatory_test_access=false`。因此当前仍不得启动 MedTAWS、权重训练、confirmatory、sealed
test 或 additional 271；若要继续，必须另立并冻结对应协议。

---

## 50. 2026-07-21 21:01:37（北京时间）：官方 validation-protocol A0 对照

### 50.1 为什么新增 A0

逐行核验 `SAM-Med3D-main/validation.py` 后确认：A1 的 click 1-2 RITM、click 3-5 uniform XOR、
current sparse point only、exact decoder-lowres recurrence 与官方主干一致；主要差异不是点击算法，
而是官方使用 GT label-centered crop、crop-only Dice、单模态输入和 max-over-click。官方代码还有
一个必须忠实记录的阈值差异：下一点击使用 raw logit `>0.5`，计分使用 sigmoid `>0.5`（raw
logit `>0`）。

因此新增独立 A0 reproduction control，不覆盖 A1。A0 使用 turbo checkpoint、FP32、seed 2023、
四个独立单通道 modality arms；只访问 184-case / 84-subject development tune。协议：
`workspace/BraTS/training_protocols/brats2024_sammed3d_official_a0_reproduction_v1.md`。

### 50.2 完整结果

| Modality | exact click-5 D6 | official best-of-five D6 | exact click-5 D3 | official best-of-five D3 |
|---|---:|---:|---:|---:|
| T1n | 0.351904 | 0.375347 | 0.359201 | 0.382563 |
| T1c | 0.393888 | 0.415088 | 0.437141 | 0.459077 |
| T2w | 0.397107 | 0.417798 | 0.376833 | 0.400866 |
| T2f | **0.470783** | **0.490891** | **0.474259** | **0.496426** |

T2f official D6 比 A1 `0.416978` 高 `+0.073913`，但仍不是论文声称的 `0.60-0.73`。T2f
SNFH/WT 为 `0.711023/0.745269`，T1c 更适合 ET/TC、T2w 更适合 RC。post-hoc
per-target best-modality oracle D6 `0.554994` 只能用于提出 target-conditioned routing/fusion
假设，不得作为结果或直接解封 confirmatory。

### 50.3 artifact、完整性与权限

canonical aggregate：
`workspace/BraTS/prompt_binary_official_a0/a0_official_seed2023_strict_tune184_aggregate.json`  
SHA-256：`ef08b4b957a06ca6bdedd261f8b5b32c08708251604f59ef36f7593075fdfbba`

184 cases / 84 subjects；present counts `129/72/147/184/130/184`；四个 modality arms 共 3384
evaluated records；无 nonfinite、threshold drift 或 model-state drift。`1865/3384` 条 exact click-5
低于 earlier best；112 条 modality-query crop coverage 不完整且官方 scorer 忽略窗外 GT。定向测试
`17 passed`。

A0 仅是 oracle reproduction diagnostic，`automatic=false`。confirmatory、sealed test、additional
271 全部未访问；本协议不授权 MedTAWS 或 weight training。下一步若做 target-conditioned fusion
search，必须另建固定候选空间/预算协议，并先在 tune 冻结选择，再申请一次 confirmatory 对照。

---

## 51. 2026-07-22：Paper1913 对齐、P0 自动基线工程与双 B200 pilot

### 51.1 方向纠正

用户指定的 MICCAI 2025 Paper1913 是四模态、全监督、自动 3D segmentation，不包含 SAM-Med3D、
point prompt、click 或 frozen checkpoint。其 BraTS2024 D3 WT/TC/ET mean Dice 为：3D U-Net
`0.8388`、nnU-Net `0.8588`、Proposed `0.8745`。当前 A0/A1 是 oracle-interactive binary
region-query，只能放 appendix，不能再作为 automatic baseline。

官方页面标明 no code/no supplement。二作于论文后上传的仓库 README 明称 partial
implementation，无 LICENSE、weights、release 或完整依赖；实测 `128^3` forward 在 attention
shape concat 失败，loss 的 3-channel target 与 512-channel feature 也不兼容。不得复制该仓库代码
或声称忠实复现。详细审计：
`workspace/BraTS/research_audits/miccai2025_paper1913_local_gap_audit_20260722.md`，SHA-256
`0eb9e0bcbfd16d1d4422af0f85ee75bc623624d5671a4e54273488a972d59e0f`。

### 51.2 新工程

- `sammed3d/paper1913_baseline.py`：独立 P0 four-channel 3D U-Net，five-class atomic logits；
- `TumorCenteredPatchConfig`：仅训练期使用 tumor-union centroid `128^3`，验证 API 不接受该参数；
- `paper1913_voxel_d3_case_metrics`：WT/TC/ET conventional voxel Dice/HD95，明确 empty policy；
- `scripts/train_paper1913_p0.py`：manifest hash binding、BF16、CE/Composite、resume、checkpoint、
  label-free full-grid sliding-window validation；
- `scripts/audit_paper1913_p0_overfit.py`：bounded fit-crop gate；
- protocol：`workspace/BraTS/training_protocols/paper1913_aligned_automatic_p0_v1.md`，SHA-256
  `f1684d6566ed4dbf55f7454d0ba1b486eaeff6474e5cc962e3ff6e5283a4028e`。

完整测试 `434 passed`；最终定向回归 `31 passed`；`py_compile` 和 `git diff --check` 通过。

### 51.3 双 B200 数值

两张 B200 均恢复 Torch `2.12.0+cu130`、SM100、cuDNN 92000 overlay。跨 subject、四原子类
present 的 cases `00009-100/00020-100`，500-step crop gate：CE D3 `0.889378`，Composite D3
`0.895191`，两臂均 pass，证明数据/标签/模型/损失链路可学习。

同 seed 累计 764 optimizer steps、固定前 8 个 tune cases、无 GT full-grid：

| Arm | WT | TC | ET | D3 | D6 lesion-wise |
|---|---:|---:|---:|---:|---:|
| CE | 0.770658 | 0.019193 | 0.019409 | 0.269754 | 0.192669 |
| Composite | 0.735078 | 0.400264 | 0.403900 | 0.513081 | 0.214819 |

Composite 在短预算显著恢复 TC/ET，晋级正式长训工程臂；CE 作为论文文字对齐弱基线保留。D6
被 absent NETC credit 抬高，禁止只报 macro。compact artifact：
`workspace/BraTS/paper1913/p0_pilot_compact_v1.json`。
compact SHA-256：`da3282d5199563ffd5efa7f86ec2ebac5b548d22558b6078c34f5b612a0354e8`。

这些是 8-case engineering pilot，不是 184-case tune、Paper1913 reproduction 或最终 baseline。
resume 后 shuffle 重启，因此准确称累计 764 optimizer steps，不称完整 epoch。133 confirmatory、
consumed test、additional labels 均未访问。

### 51.4 下一次唯一顺序

1. 冻结正式 P0-Composite 100-epoch schedule、validation cadence、selection metric 与 loader 吞吐；
2. 从零在 764 fit 训练 P0-Composite，184 tune 做完整无 GT full-grid D3+D6；CE 只需固定弱对照；
3. P0 有效后依次做 H1 separate modality streams/local self-attention、H2 DRL+RCL(T=0.07)、
   H3 local cross-attention 配对消融；论文缺失参数必须标为 reconstruction assumption；
4. 自动 checkpoint 与 architecture 冻结后才恢复 MedTAWS，仍禁止 threshold/TTA/morphology/
   postprocess 搜索；
5. 最终配置先冻结后，才允许 948 refit -> 133 confirmatory 一次；test 继续封存。

当前两个 pilot 训练/评估进程已结束。本轮临时开启的第二张 B200 worker 已 graceful exit，只保留
原有 allocation，无额外 background 训练或评估 terminal。

---

## 52. 2026-07-22 13:10 CST：四模态 P0/P0+ 正式长训启动

### 52.1 长程目标与权限

用户本轮明确授权正式 fit/tune 长训并要求未达标持续工作。目标冻结为完整 184-case
development_tune conventional D3 `>=0.80`，优先达到/超过 Paper1913 3D U-Net `0.8388`，高分
目标 `0.8588`。必须同时报告 all-case 与 present-only WT/TC/ET、D6、HD95、D6 FP lesions。
旧 manifest 的 `authorizes_formal_training=false` 仅被本次用户授权在 development fit/tune 边界
内覆盖；133 confirmatory 与 test 仍未授权、未访问。

长程 goal thread：`019f6f09-3e56-7b91-9041-0a7b67e33a0e`。MedTAWS 只能在 baseline、权重、
推理和统计协议冻结后，以等预算与 greedy/softmax/uniform 对照恢复；不得提前用搜索掩盖无效
baseline。

### 52.2 长训前工程冻结

- 修复 `persistent_workers` 导致 epoch crop 不更新：worker 每 epoch 重建；
- `latest.pt` 每 epoch 原子保存，严格绑定 manifest/loss/sampler/model/lr/wd/seed/AMP/SW 配置；
- resume 恢复 optimizer、history、best scores、DataLoader generator 与 Python/NumPy/Torch/CUDA RNG；
- 同时保存 `best_d3.pt`、`best_d6.pt`，主 `best.pt` 按 D3；
- 增加 present-only D3 和逐区 D6 false-positive lesion 总数；
- SW batch 从 1 冻结为 8，一例 B200 smoke 峰值 `7.073 GiB`；同配置 resume smoke pass；
- 完整 suite `435 passed`，`py_compile` 与 `git diff --check` pass。

正式协议：
`workspace/BraTS/training_protocols/paper1913_p0_formal_goal_v2.md`，SHA-256
`15cbc6a7bb4d57672aaad43021766dd96266cf112a7c68408f6afb4a41c3a55a`。trainer SHA-256
`cc307435628b922b99928515419d0b4367d6382e9a1f8628d418834c0d6974a0`；loader SHA-256
`2fc3b84c579f17cf6f4f96f986fa83d43f71c1d6dc0fa693926037be13a92c9e`。

### 52.3 正在运行的正式双臂

共同配置：764 fit / 完整 184 tune、Composite、100 epochs、batch 1、BF16、AdamW
`1e-4/1e-5`、seed `20260722`、channels `24/48/96/192/320`、patch `128^3`、workers 4、完整
label-free full-grid tune 每 10 epochs、overlap 0.5、SW batch 8、D3 checkpoint selection。

| Worker | Arm | Output |
|---|---|---|
| B200 `1007515` | P0 tumor-centroid, jitter 8 | `workspace/BraTS/paper1913/p0_composite_tumor_fit764_tune184_e100_s20260722_v2` |
| B200 `1007672` | P0+ label-balanced, foreground 0.70 | `workspace/BraTS/paper1913/p0plus_composite_balanced070_fit764_tune184_e100_s20260722_v2` |

2026-07-22 13:11 CST 已用 `nvidia-smi` 确认两卡各有一个训练进程、显存约 `3498 MiB`。正式命令
不含 `--max-steps`、`--train-max-cases` 或 `--val-max-cases`。若连接中断，只能用原命令附加
`--resume OUT/latest.pt`；不得修改配置后续接同一目录。

13:16 CST 两臂 epoch 1/100 均已原子落盘 `latest.pt/progress.json`，global step 均为 764。
P0/P0+ train loss=`2.209684/2.224704`，crop foreground min=`803/33`；逐项比较 run_config 后除
`sampling_strategy` 外完全一致。首次正式完整 184-case 指标在 epoch 10 产生。

13:55 CST，P0 tumor-centroid 的首次正式 epoch-10/full-184 milestone 已落盘：all-case D3
`0.624460`，WT/TC/ET=`0.851969/0.509812/0.511597`；present-only D3=`0.726572`，其中
TC/ET=`0.660042/0.667704`（present counts `130/129`）；D6=`0.362550`。六区 FP lesions 总数
3975，密度 `F=3.60054`，逐区 ET/NETC/RC/SNFH/TC/WT=`599/299/585/1084/591/817`。
54 个 TC-absent cases 只有 8 个没有 TC FP，55 个 ET-absent cases 只有 8 个没有 ET FP，说明
epoch-10 的主要瓶颈是 absent-case 假阳性而不是 present-case 完全未学会。P0 已继续训练；该早期
结果低于 `0.80` 但不触发中止，等待 P0+ epoch 10 和后续 20..100 曲线。

14:15 CST，P0+ balanced-0.70 epoch-10/full-184 也已落盘：all-case D3=`0.610302`，
WT/TC/ET=`0.849685/0.489144/0.492078`；present-only D3=`0.711927`；D6=`0.361959`；FP
lesions 总数 3995，`F=3.61866`。相对 P0，all/present D3 分别低 `0.01416/0.01464`，D6 仅低
`0.00059`；只在 RC 上明显改善（D6 +`0.06851`、FP -172），其余五区总体退化。两臂 D3 差
不超过 0.02，但 P0+ 未达到预注册的 FP 至少降低 25% 条件，因此 epoch 10 暂以论文忠实 P0 为
主臂；两臂仍继续 100 epochs，不提前切 MedTAWS。

14:36 CST，P0 epoch-20/full-184：D3=`0.647468`（较 e10 +`0.023008`），
WT/TC/ET=`0.858391/0.540770/0.543242`；present-only D3=`0.748068`，TC/ET=
`0.688475/0.697338`；D6=`0.366376`。FP 总数 3803，`F=3.44475`，较 e10 的 3.60054 下降。
曲线仍上升，不能中止；但按线性外推不能保证 100 epoch 达到 0.80，需保留预注册的单因素救援。
e20 `best_d3.pt` SHA-256 `ca2435483ab2706a407ddb5e6b4d1b14e12c9f53aaf409d7d0cb1eab49633373`。

### 52.4 失败续研规则

100 epochs 前不以中间低分判定模型上限。若两臂最终均低于 `0.80`，先根据完整 184 曲线、
present-only D3 与 FP lesions 选择获胜 sampler，再一次只改变 loss/augmentation/model capacity
中的一个因素做成对实验。达标并冻结前不访问 confirmatory/test，也不启动 MedTAWS。

未见 full-184 结果前已预注册的具体触发树位于
`workspace/BraTS/research_audits/paper1913_p0_contingency_audit_20260722.md`：FP 密度 `F>4`
触发 Tversky `alpha=0.7/beta=0.3` 单因素；WT 已学会但 TC/ET 欠学触发 gamma-2 focal CE；广泛
underfit 才以 MONAI SegResNet 作为 architecture-only 强基线；明确过拟合才单独加三轴 flip。
这些只是后备协议，不授权提前中止当前 100 epochs 或并行偷跑新 arm。

---

## 53. 2026-07-22 14:55 CST：DPD-SAM T1ce 二值任务替换四模态主线

### 53.1 用户纠正后的可比合同

| 项目 | DPD-SAM / 当前新主线 |
|---|---|
| 输入 | 仅 T1ce，本地 `modalities.t1c`，单通道 |
| 输出 | residual tumor vs background 二值 mask |
| 提示 | 自动生成的连续 domain prompt，无人工点/框 |
| 推理几何 | 整体预处理并 resize 到 `128^3` |
| 指标 | binary DSC、IoU、HD95 |
| 重复 | 同一 patient split 的 5 个独立 seed，mean+/-std |
| 划分 | patient-level 8:2 train/test，train 内约 20% val |

因此 Paper1913 的四模态五类 P0 D3/D6 与 DPD-SAM 二值 DSC 不同任务，不能横向比较，也不能
再把前者作为 MedTAWS 的 performance 底座。

### 53.2 计算终止与保留点

14:55 CST 对两个 foreground training session 发送 Ctrl-C，随后分别登录 B200 `1007515` 和
`1007672` 执行 `nvidia-smi --query-compute-apps`，两者均无 compute process。

| 历史 arm | last atomic epoch/step | last full validation | 状态 |
|---|---:|---:|---|
| P0 tumor | 29 / 22156 | e20 D3 0.647468 | terminated, no resume |
| P0+ balanced | 17 / 12988 | e10 D3 0.610302 | terminated, no resume |

latest SHA-256 分别为
`24d9c5ee503b07e1bd429e43c1ae7a197f361b9809dcb22cd2ab5a9ac85d4f0a` 和
`1e556a72c1817221ad9cfca364fcb6f70a6e91fb3f452ec9d10798cea5c4cfba`。旧 `progress.json`
显示 `running` 是因为 Ctrl-C 发生在 epoch 中间；同目录 termination marker 才是最终状态。

### 53.3 DPD-SAM PDF 核验

本地 PDF SHA-256：`ce49d74a94dcd3529c394a39458ea8583ac8d8226aa0d02de099112171473b4d`。
正文明确 `C=1` T1ce、single-channel continuous mask、SAM-Med3D backbone、offline K-means++
`K=4`、rank-1 domain prompt、50-epoch warm-up、200 epochs、batch 2、AdamW `4e-4`/`0.1`、
whole-volume `128^3`、TorchIO affine/elastic/Gaussian noise、diffusion T=1000/DDIM S=10、五次
seed。BraTS2024 binary DSC：U-Net `51.9+/-2.1`、nnU-Net `72.8+/-1.2`、SAM-Med3D
`74.1+/-1.5`、CDSG-SAM `83.1+/-0.8`、DPD-SAM `86.8+/-0.6`；DPD-SAM IoU
`77.0+/-0.5`、HD95 `5.0+/-0.8 mm`。

### 53.4 当前阻塞与下一步

PDF 只称 `residual tumor tissues`/single-channel ground truth，没有声明本地 BraTS 标签
0/1/2/3/4 到二值 target 的集合。ET、TC、WT 或其他 union 会显著改变 DSC，禁止猜测。当前并行
核对作者代码和本地标签证据；映射冻结后再构建 1350-case patient-level 64/16/20 split、二值
loader/evaluator 和五 seed baseline。新协议草案：
`workspace/BraTS/training_protocols/dpdsam_t1ce_binary_restart_v0.md`。confirmatory/test 的旧
权限边界不自动迁移到新 split；在新 split ledger 冻结前不得评分独立 test。

---

## 54. 2026-07-22 16:21 CST：T1ce baseline 几何纠错与自动提示工程

### 54.1 目标与 split 已冻结

本地 DPD-SAM cohort 表的 residual volume `41.64 (62.14) ml`、`<5 ml=2.8%` 与全量元数据
按 `{1,2,3}` 聚合所得 `41.641/62.141/2.81%` 精确吻合，故前景冻结为 `{1,2,3}`，RC=4 排除。
新 split 只含 T1ce path 与 label path：872 cases / 389 subjects fit，209 / 97 validation；旧 outer
269 cases / 127 subjects未打开。train/validation SHA-256 分别为
`ce922bdbf2f4bd693aac816a357e76e874aaf6f45d4fc16eeeb357b0c15557df` 和
`e225e244e1fe36944c7f33f64d5316b8665ecd2e7bbac05d2411790bf07059ce`，split manifest 为
`7447e3cb2499a9b54a74f3fc8522f32eb6301031c0bddb9c5c848313ab641117`。

### 54.2 全量几何审计发现并修复真实 bug

1081 个 development case 均为 `182x218x182`、1 mm、LAS，T1ce/seg affine mismatch=0，且
`{1,2,3}` 空目标=0；label 4 出现在 921 例并被正确排除。旧 mask downsample/restore 使用 legacy
`nearest`，与 image 的 center-aligned trilinear 不一致：target roundtrip DSC
mean/median/p05/min=`0.788933/0.799949/0.662055/0.275551`。改为 `nearest-exact` 后为
`0.924792/0.929385/0.876510/0.731407`，IoU mean `0.861328`。

修复覆盖 brain mask、GT resize 与 prediction restore，并新增 dual-grid scorer。论文只说明所有
volume resize 到 128-cubed，未说明恢复后评分；v2 因此预注册 model-grid 为论文对标 headline，
source-grid 为强制 secondary。审计报告 SHA-256
`4525c6d3ad4a15f52204d53a17b9d789434519824d2b4a50735efa472bbbe765`。所有 pre-fix 数值作废，
包括旧 `0.299948` LoRA pilot；不得用来估计修复收益。

### 54.3 模型与训练器

- B0/B1：单个 case-independent sparse target token；B1 加 encoder Q/V LoRA。仅为工程 control。
- D1：frozen encoder GAP surrogate、fit-only K-means++ K=4、两层 adapter、rank-1 dynamic domain
  prompt、mask decoder；fit artifact 明确不读 label/validation image，cluster counts `67/99/47/43`。
- A2：image-conditioned 3D spatial dense prompt + learned sparse target token + official token0；明确
  `not_dpdsam_paper`，用于检验空间定位缺失。
- B0/D1 runner 均具 manifest/checkpoint strict binding、完整 RNG/DataLoader resume、image-first
  validation、dual-grid atomic outputs 与 `best_model_grid_dsc.pth` selection。

模型层 D1/A2、runner、data/split 的完整 CPU suite 为 `479 passed`，仅有 surface-distance 的既存
SciPy deprecation warnings。

### 54.4 corrected 256/64 结果与运行状态

相同 cases/seed/augmentation/lr/wd/threshold 的 20-epoch corrected gate：

| Arm | model DSC | source DSC | model IoU | model HD95 mm | 决策 |
|---|---:|---:|---:|---:|---|
| B1 static + encoder LoRA-r4 | 0.325470 | 0.325024 | 0.209179 | 32.085 | 晋级全量 |
| D1 dynamic K4/N1, frozen encoder | 0.161892 | 0.161581 | 0.093162 | 56.324 | 不扩、不接 CDR |

D1 的结果说明只有全局 domain prefix、没有空间提示或 encoder adaptation 时 coarse mask 仍无效。
CDR 应建立在可靠 initial mask 上，因此此时接 diffusion 会混淆根因。

16:27 CST 用户要求暂停并确认指标定义。全量 B1 停在 atomic epoch 1、尚无 validation；B1
no-augmentation pilot 停在 epoch 8，最后 e5 model-grid DSC `0.202683`。两目录均写入
`PAUSED_BY_USER_20260722.json` 且 `resume_authorized=false`；两张 B200 已核验无 compute，A2 runner
接入也已中止。outer test 未访问；MedTAWS 未启动。

---

## 55. 2026-07-22 16:34 CST：1350-case 随机 subject-clean 80/20 切分

本轮只做数据 membership 切分，不复制 NIfTI、不启动训练。全量 cohort 为 1350 scans / 613
subjects；同一 subject 的全部纵向 timepoints 必须进入同一侧，避免 scan-level 随机造成泄漏。

算法固定为：Python RNG seed `20260722` 随机打乱 subject，再用 subject scan count 做 exact subset
sum，选择恰好 270 个 test scans。结果：

| Split | Scans | Subjects | Scan ratio |
|---|---:|---:|---:|
| Train | 1080 | 494 | 0.800000 |
| Sealed test | 270 | 119 | 0.200000 |

验证：case overlap=0、subject overlap=0、union=1350；全部 6750 个源路径存在；builder/test verifier
均未打开 test NIfTI；同 seed 第二次构建的 train/test JSONL 与正式输出逐字节一致。

产物：

- `data/datasets/brats2024_random_subject_80_20_seed20260722_v1/train_cases.jsonl`
  SHA-256 `aa2e55915e2a4bc7c2af2e95d6e08f2199397694fe4a92ea66999b534a22f63d`；
- `sealed_test/test_cases.jsonl` SHA-256
  `7b91690c63a5eb359be105b3cc7c4ad563d41fca2b2c84e521251b7ed006b8d2`；
- `sealed_test/SEAL.json` SHA-256
  `ea8099726c5ed6fa71df9e6d599f6c29a0c27efd862603109d1af43487b628e6`；
- `split_manifest.json` SHA-256
  `215df5594550d939707b25dc3671082ae68cb000b8c1fa9615d9bcb604c0bc25`。

构建脚本 `scripts/build_brats2024_random_train_test_split.py` SHA-256
`4b795837a966f7badcbc2630426ef0135a33e535c161881cde204d8d5ef7f491`；聚焦测试 `3 passed`。
新 split 是 target/modality-neutral 的原始路径 ledger，保留四模态与 seg 路径；后续即使只使用
T1ce，也从同一 membership 派生，不得重新抽有利 test。当前尚未从 1080 train 内划 validation，
所有 GPU 训练仍暂停。

---

## 56. 2026-07-22 16:52 CST：原始 SAM-Med3D T1ce 零训练自动 test baseline

用户明确授权后打开 `sealed_test/test_cases.jsonl` 的 270 cases。test manifest SHA-256 固定为
`7b91690c63a5eb359be105b3cc7c4ad563d41fca2b2c84e521251b7ed006b8d2`。模型严格加载原始
`vit_b_ori` checkpoint `sam_med3d_turbo.pth`（SHA-256
`899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`），全部参数冻结，无训练、
微调或 adapter。

协议：只读取 `modalities.t1c`；source 非零 T1ce population z-score；whole-volume trilinear resize
到 `128^3`；raw logit >= 0；无 TTA、后处理和参数搜索。两个 prediction arm 都不访问 label：
`unprompted` 为 points/boxes/masks 全 None；`t1ce_brain_center_point` 为 T1ce 非零脑区 bbox 中心的
单个正点。两种 prediction 均完成后才打开 seg，分别对 ET=raw 3 与 residual=raw `{1,2,3}` 计分。

| Arm | Target | model DSC mean/std | model median | source DSC mean |
|---|---|---:|---:|---:|
| unprompted | ET | 0.002272/0.005939 | 0.000000 | 0.002261 |
| unprompted | residual | 0.006127/0.007366 | 0.003758 | 0.006196 |
| T1ce center point | ET | 0.002231/0.005716 | 0.000000 | 0.002200 |
| T1ce center point | residual | 0.005656/0.007050 | 0.003221 | 0.005790 |

ET target-present 209/270，unprompted present-only model DSC `0.002936`；residual 270/270 present。
所有 arm 均在 270/270 预测非空。这是严格 automatic no-target-prompt baseline，不能与 DPD-SAM
论文 prompted SAM-Med3D `0.741` 混比。完整报告和评测器 SHA 分别为
`88995830d20ba732f37fd2f1da04c7156aaacf7f164532431b07f608263f5273`、
`609e9f1067c77c5002db7a42ee9b56d2c5802d33c8a50972bb38bf2dfdc9908a`。单卡 B200 `1007515`
执行完成且进程已退出。

---

## 58. 2026-07-22 17:18 CST：SAM-Med3D 官方接口根因纠错

实跑仓库 `phase0_reproduce.py` 的 3 个历史 `tumor_WT` 病例，当前输出 best-of-5
`0.7777/0.6111/0.7694`（mean `0.7194`），保存基线 mean `0.7051`；checkpoint、decoder、
B200 runtime 均正常。

270-case 新 test 的 corrected oracle-crop 对照：

| Arm | C1 | C2 | C3 | best-of-3 |
|---|---:|---:|---:|---:|
| accumulated uniform points, no mask | 0.008997 | 0.011974 | 0.014095 | 0.014735 |
| accumulated uniform points + recurrent mask | 0.243101 | 0.291748 | **0.330096** | 0.339958 |
| current uniform point + recurrent mask | 0.243101 | 0.285019 | 0.310956 | 0.335998 |
| upstream RITM/current point + recurrent mask | 0.257079 | 0.303683 | 0.330285 | 0.352253 |

输入/目标为 T1ce / `{1,2,3}`；预处理为 `ToCanonical`、GT-centered CropOrPad128、crop 后
TorchIO masked z-score。Crop target coverage mean/min=`0.998466/0.932624`，248/270 完整覆盖。
这是 oracle crop metric。Primary 用户协议的 C3 是 `0.330096`；旧 `0.011020` 正式撤销。

官方 post-treatment `WT={1,2,3}` 与本节 residual `{1,2,3}` 完全相同，所以本节结果同时也是
T1ce official-WT 三点击结果。历史 AFlow `tumor_WT={1,2,3,4}` 含 RC，是不同旧口径。

DPD-SAM PDF 不披露 residual raw-label mapping 与 automatic baseline prompt，故其 `0.741` 不能
由当前三点击结果宣称复现。仓库历史 `~0.716` 是 T2-FLAIR、WT `{1,2,3,4}`、5-click
best-of-5，任务不同。

报告 `workspace/BraTS/t1ce_official_geometry3_test/full_270_single_b200_v2.json` SHA-256
`18b6e245fe0a566ec7498d3f478b1d9e4405d4ae951b4b09e4031223ade37460`；评测器 SHA-256
`d4ed817832e0aca948b11a8024bf91c22bbec0fba0cf20f88f4682d604ed63c5`。根因审计记录全部
接口差异与解释边界；B200 已无 compute。

---

## 57. 2026-07-22 17:02 CST：T1ce 三次累计 GT corrective-click test

固定 residual `{1,2,3}` target、seed `20260722`、case-derived RNG。Click-1 从 GT 前景均匀采样
正点；click-2/3 从上一轮 binary prediction XOR GT 均匀采样，FN 标 1、FP 标 0；points 每轮
累计为 1/2/3。mask prompt 始终 None；decoder 为 `multimask_output=False` token0，阈值 0.5，
whole-volume `128^3`，无 TTA、postprocess 或参数搜索。

| Click | model DSC mean/std | median | model IoU | source DSC |
|---:|---:|---:|---:|---:|
| 1 | 0.008049/0.009776 | 0.005116 | 0.004065 | 0.008079 |
| 2 | 0.010133/0.011750 | 0.006519 | 0.005128 | 0.010126 |
| 3 | 0.011020/0.012282 | 0.007406 | 0.005579 | 0.010966 |

Click-2 FN/FP=`236/34`，click-3=`224/46`；click-3 相对 click-1 paired delta
`+0.002970`，improved/equal/worse=`179/15/76`。270/270 trajectories 合约核验通过。

完整报告 `workspace/BraTS/t1ce_corrective3_test/full_270_single_b200.json` SHA-256
`c495cf2626dbc7edcbeb17f2c9a72a8d16b68e2b2f633a73c4525c55637ad7c8`；脚本 SHA-256
`7e5c9a67a8ccb6a21e0bee5601b2d5e65d9ddd87936352a0e9f35f61737d1756`。该结果不是论文
`0.741` 复现；下一诊断轴是 upstream GT-centered ROI 和 recurrent low-res mask。单卡运行
`235.71 s`，现无 GPU compute。

---

## 59. 2026-07-22 18:17 CST：MedTAWS-GRPO 英文论文包

用户要求参考 `article/BCGFMamba__Bottleneck_Context_and_Gated_Fusion_for_Postoperative_Glioma_Segmentation__4_.pdf`
的篇幅和组织方式，直接撰写以 MedTAWS 与 GRPO 为核心 insight 的论文。主 agent 与三个并行
只读 agent 分别完成参考稿结构审读、MedTAWS 证据审计和 GRPO 实现/结果审计。

论文包位于 `../article/MedTAWS_GRPO/`：

- `manuscript.md`：4,203 词英文正文，SHA-256
  `68e1d856f504284392d337cf89bca423f1d10db6e50e65e51b0f5c3cac7610a4`；
- `MedTAWS_GRPO_manuscript.pdf`：8 页 review PDF，SHA-256
  `0d2d98f93b13beb7c377927a20b5b09b9d6a5a92d7c38876cc438e3c50d4f374`；
- `main.tex` / `references.bib`：可投稿源文件，SHA-256 分别为
  `015654b923e161fdd3b51845d67597dea17d725f5dea2b9ca56b4d9fc1917f99` /
  `cf41a5d4d3b6191a075058d714eb94ed1f21cd3f625acc224980d211eff5f818`；
- `figures/`：方法总览与历史开发结果图；
- `EVIDENCE_MATRIX.md`：每条 claim 的允许/禁止表述；
- `CONFIRMATORY_EXPERIMENTS.md`：最终四臂与一次性 holdout 的冻结清单。

论文准确定位：

1. MedTAWS 首次出现明确写为 built on AFlow 的 stochastic UCB-guided workflow-tree
   search；没有 rollout、标准 `N/W/Q` 和 ancestor backup，因此不包装为标准 MCTS。
2. 训练侧写为 **GRPO-style group-relative prompt-trajectory alignment**。当前 hard mask 是
   确定性阈值 action，`old_logp` 来自同次 forward 的 detach，ratio 前向恒为 1，不能写成
   标准 PPO/GRPO。
3. 历史 RC 同规模对照为 supervised118/e10 `0.57283` vs GRPO-style118/e10 `0.58932`，
   两 seed mean delta `+0.01649`。这是旧单模态 T1c、raw RC `{4}`、GT-centered crop、
   5-click oracle、subject-leaking development signal，不是当前 T1ce residual、独立 test
   或 SOTA 结果。
4. MedTAWS 历史数值只证明工程闭环和 target-specific candidate signal；没有有效等预算证据
   证明其优于 greedy。四 mask-token 被误当语义类的结果继续撤销。
5. 当前 1,350-scan corrected protocol 尚未运行 GRPO，也没有完成 base / MedTAWS /
   GRPO-style / combined 的 subject-clean 四臂。因此正文明确不声称超过外部论文。

本轮只读文档、代码和历史 artifact，未启动 GPU、未访问新 test、未运行 threshold/TTA/
postprocess 或新的训练/搜索。

---

## 60. 2026-07-22 19:55 CST：残余肿瘤定义冻结为 official WT 后重算 Dice

用户确认 residual tumor 定义为 BraTS 2024 post-treatment official WT，即 raw `{1,2,3}`。
现有 270-case evaluator 原本就用 `np.isin(seg,(1,2,3))`，因此预测和 target 完全相同，
不需要重跑 GPU。

主协议（T1ce、累计三点击、recurrent mask、oracle CropOrPad128）重聚合：

| 聚合 | C1 | C2 | C3 | best-of-3 |
|---|---:|---:|---:|---:|
| scan macro | 0.243101 | 0.291748 | **0.330096** | 0.339958 |
| 119-subject balanced | 0.242981 | 0.292463 | **0.329269** | 0.338133 |

C3 scan median/IQR 为 `0.336744 [0.196602,0.438494]`；subject-bootstrap 95% interval
为 `[0.304634,0.353668]`。把 crop 外 WT 前景全部计为 FN 后，C3 scan/subject Dice
`0.329909/0.329066`，说明 crop truncation 不是 0.33 的主因，但 GT-centered crop 仍是 oracle。

旧 `~0.7` 来自 T2-FLAIR、legacy `tumor_WT={1,2,3,4}`、5-click best-of-5，不是当前
official WT。完整审计：
`workspace/BraTS/research_audits/t1ce_official_wt_dice_reaggregation_20260722.md`。
本轮只读现有 JSON，未运行模型、未新增 test 访问、未做 threshold/TTA/postprocess。

---

## 61. 2026-07-23 12:07 CST：T1ce→WT SAM-Med3D 微调与冻结测试

按用户纠正后的任务完成 work #2 首轮正式微调。外层沿用 subject-clean 80/20：
1,350 scans / 613 subjects 按 seed `20260722` 分为 1,080 train / 270 test；再仅从
1,080 训练池按 subject、seed `20260723` 精确切出 972 train / 108 validation。内层
train/validation 为 444/50 subjects，case 和 subject overlap 均为 0。输入仅 T1ce，
二值目标为 official `WT={1,2,3}`，RC `{4}` 排除。

训练协议：

- `ToCanonical`，WT 映射后按 GT 居中 `CropOrPad(128^3)`，正体素 sample-std z-score；
- 三次累计点击：C1 从 GT 前景均匀采正点；C2/C3 从 XOR error 均匀采样，FN 正、FP 负；
- 每轮传入全部历史点和上一轮 low-resolution mask；预测固定 raw logit `>0`；
- image encoder 冻结，仅训练 prompt encoder + mask decoder，共 7,584,304 trainable /
  100,509,232 total parameters；
- mean 3-click DiceCE，BF16，batch 1，LR/WD=`1e-4/1e-4`，10 epochs；
- 只按 108-case validation final-click scan-macro Dice 选权重，无 threshold、TTA、
  postprocess 或 MedTAWS 搜索。

训练 Dice 从 epoch 1 `0.47766` 升到 epoch 10 `0.69388`；validation 最佳为 epoch 8，
trainer C3=`0.536857`。严格 reload 后独立 validation 结果：

| Weights | C1 | C2 | C3 | subject-balanced C3 |
|---|---:|---:|---:|---:|
| untouched SAM-Med3D | 0.224873 | 0.273186 | 0.307312 | 0.302611 |
| fine-tuned epoch 8 | 0.514882 | 0.530766 | **0.535888** | **0.535369** |

同 seed validation C3 absolute gain=`+0.228576`。最佳 checkpoint：
`workspace/BraTS/sammed3d_t1ce_wt/decoder_cumulative3_logit0_train972_val108_e10_s20260723_v3_isolated/sam_model_val_dice_best.pth`,
SHA-256 `4a325a6ffca148941c5d66f4802f331d13b8ddb7862da299981e8fd6be84d7c8`。

在 UTC `04:01:52` 先写入冻结配置
`workspace/BraTS/sammed3d_t1ce_wt/FROZEN_E8_CONFIGURATION_20260723.json`
（SHA-256 `2019f5a3239ce67616271879e8468b62fa09603aa70d54fe2f8f5ebe3ef5ab03`），
随后仅运行一次 270-case test：

| Aggregation | C1 | C2 | C3 |
|---|---:|---:|---:|
| scan macro | 0.530552 | 0.544826 | **0.549828** |
| 119-subject balanced | 0.513001 | 0.526905 | **0.531978** |

Test C3 median/std=`0.575590/0.157943`，IoU mean=`0.394599`。报告
`workspace/BraTS/sammed3d_t1ce_wt/FROZEN_e8_test270_cumulative3_s20260723.json`
SHA-256 `91f6ff7e9426d74e6d7fae73fe172806c1427da17d951b18614e65a9389700db`；
freeze/result 绑定记录
`workspace/BraTS/sammed3d_t1ce_wt/FINAL_TEST_RESULT_E8_20260723.json`
SHA-256 `a9a7c8af350bbdcd6680e5b2e4efd9d2eec36fc08f0710e73ba0c265cf748b68`。

证据边界：这 270 例在项目历史上已用于 untouched-weight zero-shot 评估，因此不是全局
pristine external test；但本轮训练、epoch 选择和配置冻结未使用 test labels。历史 test
zero-shot C3 `0.330096` 与本轮 `0.549828` 的差为 `+0.219732`，两次 click RNG seed
不同，只能作 unpaired diagnostic comparison；论文主训练增益应优先报告同 seed validation
对照。本轮新代码隔离于 `scripts/train_sammed3d_t1ce_wt.py` 和
`scripts/evaluate_sammed3d_t1ce_wt_checkpoint.py`；历史 trainer 字节未变，SHA-256
`0c5026e98b2e0f848c16b436610e9b7eca391995c062ac128ce1bee26d1bfa4f`。
完整测试 `501 passed`。两个早期错误协议 run 已显式移入 `INVALID_*`，不得进入结果表。
冻结 test 后未做任何调参；B200 评估进程已退出，无 GPU compute 残留。

---

## 62. 2026-07-23 13:58 CST：低学习率重训与短程精修

上一版 section 61 已完整冻结和归档，本节不覆盖其 checkpoint 或 test 结果。原训练在
epoch 8 validation 达峰、epoch 10 下降，而 train Dice 持续上升，因此问题不是单纯
“10 epochs 太少”，而是恒定 `1e-4` 下已经出现过拟合。新一轮严格只用同一
972 train / 108 validation，固定 validation seed `20260723`，未读取 270-case test。

冻结并执行的优化臂：

| Arm | Start | LR | Completed / selected | trainer val C3 |
|---|---|---:|---:|---:|
| v2a | untouched base | `3e-5` | 12 / e9 | 0.538847 |
| v2b | v1 e8 | `1e-5` | 4 / refine e1 | **0.540989** |
| v2c | v1 e8 | `3e-6` | 8 / refine e6 | 0.538723 |

v2a/v2b 采用最佳后 patience=3 early stop；任何被中断的 partial epoch 均不计入。
三个 selected-best checkpoint 严格 reload 后的同协议 independent validation：

| Weights | C1 | C2 | C3 | subject-balanced C3 |
|---|---:|---:|---:|---:|
| v1 e8 reference | 0.514882 | 0.530766 | 0.535888 | 0.535369 |
| v2a e9 | 0.502699 | 0.525440 | 0.531548 | 0.532079 |
| **v2b refine e1** | **0.522229** | **0.536232** | **0.539075** | **0.539514** |
| v2c refine e6 | 0.518749 | 0.532595 | 0.534862 | 0.535202 |

validation winner 是 v2b：
`workspace/BraTS/sammed3d_t1ce_wt/v2b_e8refine_lr1e5_e12_s20260724/sam_model_val_dice_best.pth`,
SHA-256 `2b7d5907c78ce2ea7e1b01ff393a5597289af08670f9b4cf2d6e203d2bed5015`。
它相对 v1 的 independent C3 delta=`+0.003187`，case improved/worse=`56/52`；
subject-balanced delta=`+0.004145`，27/23 subjects improve/worse。固定 seed、10,000 次
subject bootstrap 95% interval=`[-0.003220,0.011167]`，包含 0，故只能写“小幅正点估计”，
不能宣称明确优于 v1。

预注册、适应性决策和最终 validation 记录分别为：

- `workspace/BraTS/sammed3d_t1ce_wt/RETRAIN_PLAN_20260723.json`，SHA-256
  `f75ad93f23c409a3519c2dffcbd98e058150bccf3b8099e89799f1591f4eb5a2`；
- `workspace/BraTS/sammed3d_t1ce_wt/RETRAIN_ADAPTIVE_DECISION_20260723.json`，SHA-256
  `4eeb43b3594b7ee77be793461bd42772524fb42e8b11131b6a6bb8c0131fc51d`；
- `workspace/BraTS/sammed3d_t1ce_wt/RETRAIN_VALIDATION_RESULT_20260723.json`，SHA-256
  `b19f6657340486a91bacbe48435b39e18b1e093c8efbc677d7185226c31fc3ba`。

本轮结论：直接增加 epoch 不成立；当前有效 schedule 是 v1 epoch-8 后只精修一轮
`1e-5`。v2b 仅是 validation-selected candidate，尚未重新冻结或评估 test。未运行
MedTAWS、threshold、TTA、postprocess；两张 B200 均无 compute 残留。

---

## 63. 2026-07-23 14:07 CST：v2b 冻结后 270-case test

用户明确授权对 validation winner v2b 运行 270-case test。UTC `06:03:21` 先冻结
checkpoint、manifest、seed 和完整三点击协议，再运行唯一一次评估；未运行 threshold、
TTA、postprocess 或 MedTAWS 搜索。

| Aggregation | v1 C3 | v2b C1 | v2b C2 | v2b C3 | v2b-v1 C3 |
|---|---:|---:|---:|---:|---:|
| 270-scan macro | 0.549828 | 0.532414 | 0.548724 | **0.553028** | +0.003200 |
| 119-subject balanced | 0.531978 | 0.516338 | 0.532211 | **0.535621** | +0.003643 |

v2b test C3 median/std=`0.576496/0.160351`，IoU mean=`0.398168`。同 seed、同 case
配对下，C3 有 158/112 cases 和 66/53 subjects improve/worse；10,000 次 subject
bootstrap 95% interval=`[-0.002009,0.009182]`，包含 0。因此最终结论仍是“小幅正点
估计”，不能宣称统计明确优于 v1。第二、三次点击来自各模型自身 error region，故两模型
的纠错点轨迹可以不同。

冻结配置：
`workspace/BraTS/sammed3d_t1ce_wt/FROZEN_V2B_CONFIGURATION_20260723.json`
（SHA-256 `981dee6cf926f2a52edc15c3e57d9e42919673900dd79bcd7df4b31e692e14c3`）；
逐 case 报告：
`workspace/BraTS/sammed3d_t1ce_wt/FROZEN_v2b_test270_cumulative3_s20260723.json`
（SHA-256 `827a98496c4554f029471d7d6a9f25fc613a00e80f4743977e0b070af7040045`）；
最终绑定记录：
`workspace/BraTS/sammed3d_t1ce_wt/FINAL_TEST_RESULT_V2B_20260723.json`
（SHA-256 `884973d5f69ff8012089f762c54cf723ecce23346f19dc5a2f14f3f749c84bb1`）。

证据边界：test 未参与 v2b 训练或 validation 选择，且评估后不再据此调整学习率；但它已
用于 v1 和项目历史评估，不是 globally pristine external test。当前可用资源是一张
worker `1007672` 的 NVIDIA B200，本次 270-case 评估耗时约 175 秒，进程已退出。

---

## 64. 2026-07-23 14:38 CST：本地 checkpoint 保守清理

本地 `/dev/vdc` 清理前为 125G 总量、98G 已用、27G 可用（79%）。审计发现
`workspace/BraTS` 的 54G 中有 119 个 `.pth`，合计 51.084 GiB。按“保留正式
selected-best、删除可重建冗余”执行：

- 删除 23 个同目录已有 best 替代且无结果引用的 `latest.pth` /
  `sam_model_latest.pth`；
- 删除 7 个 smoke/INVALID 权重；保留其 JSON、CSV、日志和配置；
- 删除仓库内 5 个 Python/pytest cache；
- 未删除任何正式 best、当前 v2b、训练/评估报告、源代码、venv 或 `/mnt` 数据。

共删除 30 个 checkpoint、13,856,322,730 bytes（12.905 GiB）。清理后
`workspace/BraTS` 约 42G，剩余 89 个 checkpoint / 38.180 GiB；本地盘 86G 已用、
40G 可用（69%）。剩余 checkpoint 全量 SHA-256 查重没有字节级重复。

三条没有 best 替代的 DPD-SAM full-run `latest.pth` 明确保留。T1ce→WT v1/v2a/v2b/v2c
best、v2b final result 和 270-case report 的 SHA-256 均复核未变。清理记录：
`workspace/BraTS/LOCAL_STORAGE_CLEANUP_20260723.json`，SHA-256
`1474cdf75044fb9c8e13debf8d34f0e97e480c1c17f271feaf2c7b309a953671`。

---

## 65. 2026-07-23 15:10 CST：whole-volume + encoder LoRA + augmentation 启动

三路并行完成数据增强、image-encoder 策略与实验协议审计。新实现保留旧
`gt_crop/none/rank0` 复现路径，并新增 whole-volume `128^3`（image trilinear、label
nearest-exact）、单次 source-nonzero z-score，以及 train-only affine/elastic/Gaussian
noise。image encoder 首轮使用 12-block Q/V LoRA-r4/a8；prompt encoder 与 mask decoder
继续训练。

定向测试 `33 passed`，实现侧完整定向集合 `44 passed`；真实 development train/validation
数据均输出 `[1,128,128,128]`、finite、binary target，validation 无增强。A10 的
2-train/2-validation/e1 真反向 smoke 成功，train/validation C3=`0.215246/0.085975`，
checkpoint 保存完成。小样本数值仅验证工程通路；smoke best SHA-256
`5382b7b6d2826912c64f84b64cf5a4d3f82b7a0ee15e91fe1e22ea35c8018d9b`。

正式 arm 已在 worker `1009989` 启动，PID `3166`：untouched base，972 train / 108 clean
validation，50 epochs，batch 1，accumulation 2，BF16，2 workers，AdamW
LR/WD=`1e-4/0.1`，train/validation seeds=`20260725/20260723`。trainable/total parameters
=`7,731,760/100,656,688`；首次稳态显存约 6.9 GiB，epoch 1 吞吐约 1.0--1.3 case/s。

代码 SHA-256：trainer
`369986ff41166ffbbe8081b5767e1f276ac14a7ddbe03acce3b800d6ba4b0988`，preprocessing
`4d13e2b3894e2b9a91fac7f76dd74b1c744429eef665f09a5f160a604b0989ae`，launcher
`13aca363de28cbecbfe8355dcfd485d58faf33315edd325db581e01403fe116d`。

证据边界：此 arm 是 encoder PEFT，不是 full unfreeze，也不能与旧 GT-crop v2b 做单因素
因果比较。只依据 clean validation 决策；若 LoRA 无实质增益，下一步才实现独立命名、
分组学习率的 last-four-block partial-unfreeze arm。270-case test、MedTAWS、threshold、
TTA、postprocess 均未启动。

---

## 66. 2026-07-23 15:22 CST：训练结束后自动冻结并评测 270 cases

用户明确授权在当前 50-epoch FV-LoRA-Aug 训练结束后自动评测 270-case test。旧 evaluator
原本写死 GT-centered crop 且无法 strict-load LoRA checkpoint，本轮先升级为可显式选择
`whole_resize` 并通过 `peft.load_model_with_adapters` 严格恢复 adapter architecture。现有 smoke
LoRA checkpoint 已在一例 clean validation 上完成 CPU 端到端验证；报告 SHA-256
`36d25f348c8d4d3946e1c97835cb90db21b5207b16567e6060455e9876bc965e`。

fail-closed watcher 已挂在同一 A10 worker `1009989`，PID `176743`，当前状态
`waiting_for_training`。等待阶段只读取 `/proc/3166/cmdline` 和训练状态，不打开 test manifest。
训练 PID 退出后必须同时满足以下条件，否则 watcher 写 `failed` 并停止，不运行 test：

- `TRAIN_DONE` 存在，latest checkpoint 为 epoch 50，history 长度 50；
- train/validation=`972/108`，whole resize、train-only augmentation 和 test-access=false；
- encoder Q/V LoRA rank/alpha=`4/8`，validation selection metric 为 `val_dice`；
- selected-best checkpoint epoch、metadata、adapter config 与 `best.json` 严格一致；
- trainer/preprocessing hashes 与训练前 prereg 完全一致。

全部通过后，watcher 先原子写
`FROZEN_FV_LORA_AUG_TEST_CONFIGURATION_20260723.json`，其中绑定 best checkpoint SHA、
训练 artifacts、evaluator SHA、test manifest 预期 SHA、seed `20260723`、whole-volume 三点击
协议；freeze 落盘后 evaluator 才首次打开 test manifest，并只运行一次 270 cases。预定输出：

- report：`FROZEN_fv_lora_aug_test270_cumulative3_s20260723.json`；
- final binding：`FINAL_TEST_RESULT_FV_LORA_AUG_20260723.json`；
- live state：`AUTO_TEST_WATCHER_FV_LORA_AUG_20260723.json`。

当前 evaluator/watcher/launcher SHA-256 分别为
`183886e13a7a6aacbe01f9c27f7cc9860edcb081bb7b84ec24ec98fba034c02e`、
`beca6bc32e2bacb87db7f81d7045cc0f8acd370b04956dc3d931663ac2468799`、
`fbb3116b9b807278405e315ef7ea5f544b34a7acd4fb00834c6b7d58c02d3242`。相关定向测试
`24 passed`，其中 watcher/evaluator 核心测试 `5 passed`。截至 watcher 启动并确认 waiting
状态，270-case test manifest/NIfTI 均未由 watcher 打开。

---

## 67. 2026-07-24 10:36 CST：FV-LoRA-Aug 完成、SHA 勘误与 270-case 结果

### 67.1 训练终态

worker `1009989` 上的正式 arm 已完整跑完 50 epochs，训练日志包含 `TRAIN_DONE`，history
为 header + 50 rows。总耗时 `25,107.17 s`。clean validation 只按 `val_dice` 选
epoch 33=`0.5328167623`；epoch 50 train/validation C3 =
`0.6310907633/0.5235471349`。冻结 checkpoint：

`workspace/BraTS/sammed3d_t1ce_wt/fv_lora_r4_aug_train972_val108_e50_s20260725/sam_model_val_dice_best.pth`

SHA-256：
`c1d1bd79791aa7beeffc541095472dec968630cc70ea7967128c86552bd7f1ec`。
训练协议仍是 untouched base、972/108、whole-volume `128^3`、train-only
affine/elastic/Gaussian noise、12-block Q/V LoRA-r4/a8、prompt encoder + mask decoder、
三次累计 click 与 recurrent mask；不是 image-encoder base-weight full unfreeze。

### 67.2 自动 watcher 的 fail-closed 与恢复

watcher 先完成训练/selected-best/代码哈希验证，并在 UTC `2026-07-23T14:07:31` 写入原
freeze：

`workspace/BraTS/sammed3d_t1ce_wt/FROZEN_FV_LORA_AUG_TEST_CONFIGURATION_20260723.json`

SHA-256
`b80d122fbfc8cfb834ca86b143c02b3dcb1ed3b70fb23e399cd7dea990897627`。
随后 evaluator 在 byte-level manifest SHA 检查处停止：watcher 预期值误写为
`7b91690c63a5eb359be105b9cc7c4ad563d41fca2b2c84e521251b7ed006b8d2`，
而 split manifest、`SEAL.json` 和实际文件一致为
`7b91690c63a5eb359be105b3cc7c4ad563d41fca2b2c84e521251b7ed006b8d2`。
失败发生在 manifest row 解析和 test image/label 打开之前。

原 freeze 未修改或覆盖。UTC `2026-07-24T02:23:08` 先写入独立勘误：

`workspace/BraTS/sammed3d_t1ce_wt/FROZEN_FV_LORA_AUG_TEST_MANIFEST_HASH_CORRECTION_20260724.json`

SHA-256
`cbdbbf90ee68e00cebeb81548b334b543efe494b16bb4fd3d557e42c37e01bb7`。
它绑定原 freeze、split manifest、SEAL、实际 manifest、checkpoint 和 evaluator 哈希；
唯一修正是预期 manifest SHA 的一位字符，模型、epoch、seed、geometry、click、threshold
和输出协议均不变。

### 67.3 冻结 test 结果

在勘误落盘后，用 epoch-33 checkpoint 唯一运行 270-case test。任务为 T1ce、binary
`WT={1,2,3}`、RC `{4}` 排除，whole-volume resize、3 accumulated corrective clicks、
recurrent low-resolution mask、raw logit `>0`、seed `20260723`。

| Aggregation | C1 | C2 | C3 |
|---|---:|---:|---:|
| 270-scan macro | 0.508103 | 0.516545 | **0.523030** |
| 119-subject balanced | 0.494241 | 0.503163 | **0.509143** |

C3 scan median/std=`0.540526/0.177963`，IoU mean=`0.373047`。C1→C3 scan gain
`+0.014927`；test C3 相对 clean-validation selection value 为 `-0.009787`。相对旧
GT-crop v2b test `0.553028` 为 `-0.029998`，但 geometry、初始化和训练 recipe 同时变化，
不能据此归因某个单因素。当前结论是该 whole-volume LoRA+augmentation arm 没有达到
`0.7`，也没有超过项目内 v2b headline。

正式报告：
`workspace/BraTS/sammed3d_t1ce_wt/FROZEN_fv_lora_aug_test270_cumulative3_s20260723.json`
（SHA-256 `a087c357bbc71c60bba63cf179692266d2ebe19c64f58b44d8821986ae8ab13b`）。
最终 binding：
`workspace/BraTS/sammed3d_t1ce_wt/FINAL_TEST_RESULT_FV_LORA_AUG_20260723.json`
（SHA-256 `ae0f636f6246bb9ad4920b28bd3feefd568c56be4aa200812f131b3461c446f4`）。
恢复审计：
`workspace/BraTS/sammed3d_t1ce_wt/AUTO_TEST_RECOVERY_FV_LORA_AUG_20260724.json`
（SHA-256 `f41c7b653e51b6526399ee14c88ca90d8e723559e1b4f4314b842cc364e26a2d`）。

正式 report 校验了 270 unique cases、119 subjects、每例 3 clicks、所有 DSC/IoU finite 且
在 `[0,1]`，并严格匹配 manifest/checkpoint SHA、seed 和 geometry。A10 评估耗时
`388.14 s`，观测显存约 `2.49 GiB`；评估进程已退出。270-case test 已再次打开，不得用该
结果继续调当前模型。后续方法开发必须回到 train/clean-validation；本轮没有运行 MedTAWS、
threshold、TTA 或 postprocess。
