# AFlow / BraTS 2024 SAM-Med3D — 交接文档 20260807

接手者请先读完 §0 和 §7，再动任何东西。
上一份交接是 `AFLOW_BRATS_CLAUDE_HANDOFF_20260805.md`，本文件覆盖 8/05 之后到 8/07 的全部工作。

---

## 0. 三十秒摘要

**目标**：把 SAM-Med3D 在 BraTS 2024 术后胶质瘤（T1ce 单模态、WT={1,2,3}）上做到 Dice ≈ 0.73（对标 DPD-SAM 论文里报的 SAM-Med3D baseline 74.1）。

**当前最好（270 例 sealed test，已冻结发布）**：`0.6254` scan-macro C3
**当前最好（108 例 validation，未冻结）**：`0.6357`（两阶段级联，未训练，纯组合现成模型）

**离目标还差约 0.075。本次会话把 test 从 0.5711 推到 0.6254（+0.0543），靠的是一个预处理改动。**

**最重要的结论**：主瓶颈是**泛化**，不是容量、不是分辨率、不是优化配方。
容量探针证明架构在 48 例不用泛化时能拟合到 0.885，而 972 例的 validation 只有 0.61-0.64。
**"更小的模型 / 更好的 LR 配方"这条路已被四次实验证伪，不要再走。**

---

## 1. 环境（先配好，否则什么都跑不起来）

```bash
ROOT=/mlx_devbox/users/songjunyi.666/playground/AFlow
PY=/mlx_devbox/users/songjunyi.666/playground/sammed3d-venv/bin/python   # 不是 /usr/bin/python
export PYTHONPATH=workspace/BraTS/env_shims                              # 必须，见下
```

| 坑 | 说明 |
|---|---|
| **解释器** | 必须用 `sammed3d-venv/bin/python`。`/usr/bin/python` 缺 `surface_distance` 等依赖 |
| **PYTHONPATH shim** | torch≥2.6 的 `torch.load(weights_only=True)` 会拒绝 checkpoint 里的 `pathlib.PosixPath`。`workspace/BraTS/env_shims/sitecustomize.py` 把它加进 safe globals。**不要改 `sammed3d/peft.py`**——它被已完成臂的哈希绑定 |
| **`/tmp` 是易失的** | 本次会话机器被重新调度过两次，`/tmp` 被清空、设备名都变了（`nvme0n1p3`→`nvme0n1p1`、`/rootfs`→`/dev/vdbh`）。GPU 也曾整体消失过一次 |
| **根分区快满** | `/` 只剩约 6 GB。**缓存要放 `/tmp`**（本地 NVMe，1.4T+，且读写快 2.7 倍：2.4 GB/s vs 889 MB/s） |
| **用户明确要求** | **"什么都不删"**。仓库里大量 modified/deleted/untracked 文件不是清理对象。禁止 `git reset` / `git clean` |
| **CPU 争抢** | 建缓存用 56-64 进程会占满 256 核。**建缓存时不要同时跑别的 CPU 重活**，我因此浪费了两次超时 |

---

## 2. 数据与划分（冻结，不要改）

```
1350 scans / 613 subjects
  └─ 外层 subject-clean 划分 seed 20260722 → 1080 train+val / 270 TEST(封存)
       └─ 内层划分 seed 20260723 → 972 train / 108 validation
```

| 项 | 值 |
|---|---|
| train manifest | `data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/train_t1ce_wt.jsonl` |
| train manifest SHA | `ea21e2306321b71d821146ed01198f52b612d390c0736f8703f81a259c06c23e` |
| validation manifest | 同目录 `validation_t1ce_wt.jsonl` |
| validation manifest SHA | `fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c` |
| **test manifest（封存）** | `data/datasets/brats2024_random_subject_80_20_seed20260722_v1/sealed_test/test_cases.jsonl` |
| **test manifest SHA** | `7b91690c63a5eb359be105b3cc7c4ad563d41fca2b2c84e521251b7ed006b8d2` |
| test 规模 | 270 cases / 119 subjects |

**目标定义已被验证与 DPD-SAM 一致**（972 例上比对其 Table I）：

| | median (ml) | IQR | STR率(<5ml) |
|---|---|---|---|
| DPD-SAM 报的 | 41.64 | 62.14 | 2.8% |
| 我们 WT={1,2,3} | **41.84** | **61.26** | **3.0%** |

其他候选定义（TC/ET/NETC/SNFH/RC/ALL）全部差得离谱。**所以 0.6254 vs 0.741 的差距是真差距，不是任务不同造成的。**

---

## 3. 协议（冻结）

```
输入      T1ce 单通道
目标      WT = 原始标签 {1,2,3}；RC 标签 {4} 排除
几何      128^3（SAM-Med3D 的位置编码钉死在 8^3 token 网格，输入尺寸不可改）
点击      3 次累加纠正点击；click1 在 GT 前景均匀采样，click2/3 在 XOR 错误区采样
          FN→正点，FP→负点；全部点累加；上一步低分辨率 mask 循环喂回
阈值      raw logit > 0
seed      validation/test 评估 20260723；训练 20260805
聚合      同时报 scan-macro 与 subject-balanced
```

---

## 4. 已完成的臂（全部结果）

### 4.1 test 上的项目全历史（同一批 270 例）

| 臂 | scan C3 | 备注 |
|---|---|---|
| FV-LoRA-Aug | 0.5230 | |
| v1 e8 (gt_crop) | 0.5498 | ⚠️ oracle 定位 |
| v2b (gt_crop) | 0.5530 | ⚠️ oracle 定位 |
| 全解冻·整头 resize | 0.5711 | |
| **全解冻·裁脑** | **0.6254** | ← **当前最好，已冻结发布** |

### 4.2 validation 108 上的全部臂（3击，scan macro）

| 配置 | val C3 | checkpoint SHA(前16) | 说明 |
|---|---|---|---|
| 整头 resize 全解冻 | 0.5633 | `75dd0b55f9fb29f5` | e24/64 |
| 裁脑 全解冻 | 0.6060 | `87dad82c67151128` | e78/118，**stage-1 用这个** |
| **两阶段 cascade** | **0.6357** | 组合上下两个 | **未训练，见 §5** |
| 两阶段 oracle（作弊） | 0.6380 | — | 定位用真值 |
| gt_crop 全解冻 | 0.6381 | `42f96a0a2ade9d98` | e48/100，**stage-2 用这个** |
| gt_crop last4 | 0.6274 | `f999e12432901e9f` | e24/84，**阴性** |

⚠️ `gt_crop` 几何**用真值放裁剪框**（`uses_gt_for_localization: true`），单独使用**不能**和自动方法比。它只能作为两阶段的 stage-2。

---

## 5. 本次会话的两个主要成果

### 5.1 裁脑预处理（已落地到 test，+0.0543）

**改动**：在缩到 128³ 之前，先裁到非零体素的外接框。仅此一项，其余 15 项逐项不变。

**动机（可测）**：
```
脑 bbox 只占原生数组的 47.8%，真正非零的脑组织只占 20.8%
→ 每个 128³ 输入有 79% 的体素是字面意义的 0
→ encoder 的 8^3 = 512 个 token 里约 267 个盖的是背景
```
裁脑后病灶在模型网格上的体素数 16,522 → 36,649（中位），等效间距 1.42/1.70/1.42 → 1.06/1.38/1.12 mm。

**结果**：validation +0.0427，**test +0.0543**（232/270 例变好，95% CI [+0.0455, +0.0629]，p=3.9e-35）。

- 预注册 `BRAINCROP_PREREG_20260806.json` SHA `8e52585b7d35c0b6...`
- 冻结 `FROZEN_BRAINCROP_TEST_CONFIGURATION_20260806.json` SHA `e51493aa0d747f8a...`
- 最终 `FINAL_TEST_RESULT_BRAINCROP_20260806.json` SHA `71af38c8e2436f87...`
- builder `scripts/build_t1ce_wt_braincrop_cache.py` SHA `fcb1428300be5b37...`

### 5.2 两阶段级联（validation +0.0297，**尚未冻结、尚未测 test**）

**关键洞察**：现成的 `gt_crop` 臂本身就是"定位完美的 stage-2 精修器"。把裁剪中心从真值换成 stage-1 的预测，**一行训练都不用跑**就能读出整个级联的水平。

```
stage 1 : 裁脑 checkpoint (87dad82c...) → 128³ mask → 还原到原生 1mm
crop    : tio.CropOrPad(mask_name=预测) 128³ @ 原生分辨率
stage 2 : gt_crop checkpoint (42f96a0a...) → 3 次点击
```

| | val C3 |
|---|---|
| 裁脑单阶段 | 0.6060 |
| **两阶段 cascade** | **0.6357** |
| 两阶段 oracle | 0.6380 |
| **定位不完美的代价** | **−0.0023** |

配对：+0.0297，95% CI [+0.0173, +0.0425]，82/108 变好，p=6.1e-08。小病灶收益最大（+0.053）。

**脚本**：`scripts/probe_two_stage_cascade.py` SHA `a0a0d60f49215088...`
**记录**：`TWO_STAGE_CASCADE_PROBE_FIXED_20260807.json` SHA `a092155b46269133...`

⚠️ 同目录下 `TWO_STAGE_CASCADE_PROBE_20260807.json`(0.4119) 和 `..._LCC_...json`(0.3559) 是**错误结果**，保留作证据，**不要引用**。原因见 §8。

---

## 6. 阴性结果 —— 不要重复走这些路

### 6.1 砍容量无效（`NEGATIVE_RESULT_LAST4_CAPACITY_20260807.json`）

`--unfreeze full → last4`（100.5M → 36.0M，砍 64%），gt_crop 几何：

| | val C3 | train−val gap@best |
|---|---|---|
| 全解冻 | **0.6381** | +0.1323 |
| last4 | 0.6274 | **+0.0817** |
| delta | **−0.0107** | 过拟合减少 38% |

95% CI [−0.0191, −0.0016]，79/108 变差，p=1.6e-06。

**"过拟合确实被压住了，但 val 也跟着降"** —— 被冻掉的参数不是纯冗余。
**本项目已试过 LoRA、rank-0、last-4 部分解冻、stage-2 几何上的 last-4，全解冻赢了全部四个。**

### 6.2 LR 退火无效

- gt_crop 臂：milestones [60,90] 两次都触发，最好点在 **e48**，早于两者
- last4 臂：e60 触发退火，之后 24 轮无新高，最好点在 **e24**

**两个独立臂都是如此。此前"退火值 +0.01~0.03"的估计已正式撤回。**
（限定：裁脑几何上退火从未真正触发过——早停 e118 先于里程碑 120，那里是"未验证"而非"已证无效"。）

### 6.3 交互点击基本失效

10 次 GT 纠正点击只比 1 次多 +0.039。模型已退化为对 prompt 不敏感的语义分割器。
**增加点击预算不是杠杆。**

### 6.4 输出分辨率不是瓶颈

encoder 128³→8³ token，mask decoder 输出 32³ 再 4× 上采样。
天真估计（阈值 0.5）给出天花板 0.81，但**对 32³ logit 场做梯度优化，可达 Dice 0.9266**。分辨率不是约束。

### 6.5 评分网格差异可忽略

我们的 DSC 在 128³ 网格上算，nnU-Net 类流程在原生 1mm 上算。
**实测**：把预测还原到 182×218×182 重算，0.5633 → 0.5619，**只差 0.0014**。
（GT-only 往返上限 0.9230，但那约束的是完美预测；0.56 精度下模型自身误差吸收了它。）
**这个差异既不能解释差距，也不算我们占便宜。**

---

## 7. 硬性约束（继承自 8/05 交接，仍然有效）

- **270 例 test 已被消费 6 次。** 训练/选点全程未用 test。
  **test 结果禁止用于选臂、调参、选 epoch。** 方法开发必须回到 972/108。
- 每次开封 test 前**必须**先把冻结配置写盘（含全部哈希），并在打开任何 test NIfTI 之前校验通过。`runs_allowed: 1`。
- **禁止** threshold / TTA / morphology / connected-components / 后处理搜索。
- **禁止**为了让校验通过而改写绑定哈希 —— **fail closed**。
- **禁止** `git reset` / `git clean` / 批量删除。用户明确说过"什么都不删"。
- 以下文件被已完成臂的哈希绑定，**不要修改**（要加功能就新建文件）：
  - `sammed3d/peft.py`
  - `scripts/build_t1ce_wt_cache.py` (`c3171ffc...`)
  - `scripts/evaluate_sammed3d_t1ce_wt_cached.py` (`b3acbb83...`)
  - `scripts/train_sammed3d_t1ce_wt_dpdsam_aligned.py`

---

## 8. 我（上一任）犯过的错 —— 请据此校准对本文件的信任度

**这一节请认真读。本次会话我判断失误 6 次，其中 3 次差点导致放弃正确方向。**

1. **预测裁脑会掉分** → 反了，涨 +0.0274。用户坚持先测几何是对的。
2. **说"评分网格差异对我们有利约 0.03"** → 用 `0.571×0.923` 乘出来的，**乘法无效**。实测只差 0.0014。
3. **用"预测框×乘性余量"测包含率 → 判两阶段不可行** → **测试设定就是错的**，两阶段用固定 128mm 框居中。
4. **怀疑 gt_crop 靠截断真值刷分** → 错，实测保留 99.9%。
5. **上"最大连通域"定心当修复** → 更差。WT 有 43% 多灶，扔掉了真实病灶。
6. **两阶段探针报 −0.22** → **是我的朝向 bug**：canonical 的预测配了 raw affine，`ToCanonical` 又转了一遍，框定位到镜像位置。修正后是 **−0.0023**。

**救回来的方式**：一次独立的纯几何测量给出保留率 0.9976，和探针的 0.7596 对不上 —— **矛盾暴露了 bug**。
**教训：任何结论都用第二种独立方法交叉验证一遍，尤其是负面结论。**

预测记录也保存在各 prereg 的 `agent_prediction_record` / `prediction_registered_before_the_run` 字段里。

---

## 9. 立刻可做的事（按性价比排序）

### 9.1 【推荐，已到手】把两阶段级联冻结并测 test

+0.0297 已经在 validation 上确认，**不需要任何训练**。预计 test 落在 **0.65 附近**。

步骤：
1. 写冻结配置（含 stage1/stage2 两个 checkpoint SHA、级联规则、`runs_allowed:1`）
2. 建 test 的裁脑缓存（stage-1 输入）
3. 跑 `scripts/probe_two_stage_cascade.py` 的 test 版本
4. 这会是第 **7** 次消费 test

⚠️ 需要用户明确授权 —— 用户 8/05 和 8/06 各授权过一次开封。

### 9.2 剩下唯一有希望的方向：**增加信息**

配方类杠杆已经穷尽（§6）。剩下的：

| 方向 | 理由 |
|---|---|
| 用满 1080 做 subject-clean 交叉验证 | 现在扔了 108 例只为选点；+11% 数据 |
| 更强 / 更匹配真实变异的增广 | 泛化差距是主瓶颈，而增广是唯一直接对症的 |
| 多模态输入（+FLAIR/T2） | WT 的主体是 SNFH，本来就是 FLAIR 上定义的。**但 DPD-SAM 也只用 T1ce**（论文明确写 C=1），加模态会失去可比性 |
| 换骨干（不受 128³ 约束的） | SAM-Med3D 位置编码钉死输入尺寸，是硬约束 |

### 9.3 不要做的

- ❌ 再试更小的模型 / LoRA / 部分解冻（§6.1，四次全输）
- ❌ 再调 LR 调度（§6.2，两臂证明无效）
- ❌ 增加点击预算（§6.3）
- ❌ 试图提高输出分辨率（§6.4，不是瓶颈）
- ❌ 用评分网格差异解释与论文的差距（§6.5）

---

## 10. 关于 0.73 这个目标的诚实评估

**我不认为再调配方能补上剩下的 0.075。** 理由：

1. 配方类杠杆已全部证伪（§6）
2. 两阶段能拿到手的 +0.03 已经量出来了，oracle 上界是 +0.032，没有余量
3. 主瓶颈是 972 例数据 vs 1 亿参数的泛化差距，容量探针把这一点钉死了

同时要记住 **0.741 这个参考值本身不稳**：
- DPD-SAM 在**它自己的 8:2 划分**上测的，不是这 270 例
- 它的 SAM-Med3D baseline **prompt 协议全文未披露**
- BCGFMamba 把同一个数抄成 **0.731**，与 DPD-SAM 自己的表对不上
- DPD-SAM 自己的 86.8 靠的是在 128³ 全分辨率上跑的**扩散精修 U-Net**，headline 数字不来自 SAM decoder

**任务定义、模态、几何、损失、增广都已验证对齐**（§2），所以差距是真的，但"复现 74.1"这个目标的可达性本身存疑。

---

## 11. 文件索引

### 本次会话新增的脚本
| 文件 | SHA-256(前16) | 用途 |
|---|---|---|
| `scripts/build_t1ce_wt_braincrop_cache.py` | `fcb1428300be5b37` | 裁脑几何缓存 builder |
| `scripts/evaluate_sammed3d_t1ce_wt_braincrop.py` | `f7eed32b1395190a` | 裁脑评估器包装（只改几何描述） |
| `scripts/make_braincrop_prereg.py` | `cc11d78664e3a98f` | 裁脑臂预注册生成器 |
| `scripts/make_lrdecay_prereg.py` | `02f1892d17a4352b` | LR 臂预注册生成器（**未使用**，退火已证无效） |
| `scripts/measure_stage1_localization_error.py` | `ab086187f6a74fb8` | stage-1 定位误差标定 |
| `scripts/probe_two_stage_cascade.py` | `a0a0d60f49215088` | 两阶段级联探针（**已修正朝向 bug**） |

### 关键记录
| 文件 | SHA-256(前16) |
|---|---|
| `BOTTLENECK_DIAGNOSIS_20260806.json` | `01b03395b4d126c5` |
| `BRAINCROP_PREREG_20260806.json` | `8e52585b7d35c0b6` |
| `FROZEN_BRAINCROP_TEST_CONFIGURATION_20260806.json` | `e51493aa0d747f8a` |
| `FINAL_TEST_RESULT_BRAINCROP_20260806.json` | `71af38c8e2436f87` |
| `STAGE1_LOCALIZATION_ERROR_20260807.json` | `955af24411d78df3` |
| `TWO_STAGE_CASCADE_PROBE_FIXED_20260807.json` | `a092155b46269133` |
| `NEGATIVE_RESULT_LAST4_CAPACITY_20260807.json` | `58e289c9564f9f59` |

（以上均在 `workspace/BraTS/sammed3d_t1ce_wt/`）

### Checkpoint
| 臂 | 路径（`workspace/BraTS/sammed3d_t1ce_wt/` 下） | SHA(前16) |
|---|---|---|
| stage-1 裁脑 | `sammed3d_braincrop_fullunfreeze_ddp2_train972_val108_e200_s20260806/sam_model_val_dice_best.pth` | `87dad82c67151128` |
| stage-2 gt_crop | `sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805/sam_model_val_dice_best.pth` | `42f96a0a2ade9d98` |
| 整头 resize | `sammed3d_official_fullunfreeze_ddp2_train972_val108_e200_s20260805/sam_model_val_dice_best.pth` | `75dd0b55f9fb29f5` |
| last4（阴性） | `sammed3d_stage2_gtcrop_last4_h100_train972_val108_e200_s20260807/sam_model_val_dice_best.pth` | `f999e12432901e9f` |

### 缓存（`/tmp` 易失，可用 builder 完全重建）
| 缓存 | 位置 | 状态 |
|---|---|---|
| 裁脑 train/validation | `/tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128/` | 已验证**与产出 0.6254 那次逐字节相同** |
| 裁脑 test | 同上 `/test` | 重启后已丢失，需重建（约 13 分钟） |
| gt_crop train/validation | `workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128/` | 在 `/`，完好，11 GB |
| 整头 resize train/validation | `workspace/BraTS/cache/t1ce_wt_n4_128/` | 在 `/`，完好，11 GB（test 子目录已被删） |

**重建命令示例**（约 50 分钟 / 972 例）：
```bash
PYTHONPATH=workspace/BraTS/env_shims $PY scripts/build_t1ce_wt_braincrop_cache.py \
  --manifest data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/train_t1ce_wt.jsonl \
  --expected-manifest-sha256 ea21e2306321b71d821146ed01198f52b612d390c0736f8703f81a259c06c23e \
  --expected-count 972 --split train --n4 --n4-shrink 2 --workers 56 \
  --output-dir /tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128/train
```
重建后**务必**用 `BRAINCROP_PREREG_20260806.json` 里的 `images_sha256` 核对逐字节一致 —— 已验证 N4+裁脑+resize 在换机器换硬件后完全可复现。

### 训练启动示例（H100 单卡）
```bash
AFLOW_NPROC=1 bash scripts/run_sammed3d_t1ce_wt_dpdsam_aligned.sh run \
  --cache-root /tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128 \
  --batch-size 12 --accumulation-steps 1 \
  --run-identity <新的臂名>
```
H100 显存实测：batch 12 = 65.4 GiB，batch 16 OOM。
**不要放大有效 batch**：972 例在 eff-batch 12 下每轮 81 步，翻倍就砍半；瓶颈是泛化不是优化，降低梯度噪声等于去掉一个正则化项。
