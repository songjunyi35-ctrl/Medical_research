# 已归档的历史长交接

> 这是整理前全文，正文中的“当前/唯一权威”均是历史表述，不提供恢复实验的授权。当前工作看[当前交接](../../AFLOW_BRATS_HANDOFF_CURRENT.md)。本可读副本仅调整相对链接；原始字节与哈希另存。

> 历史长交接（原文件名为8月13日，正文实际追加至8月18日）。本页保留原文追溯，不是当前执行入口；汇报看[导师报告](../../AFLOW_BRATS_WEEKLY_REPORT.md)，连续日期看[科研流水账](../../AFLOW_BRATS_RESULTS.md)，继续工作看[当前接续状态](../../AFLOW_BRATS_HANDOFF_CURRENT.md)。下文旧“唯一权威/当前”均是历史时点。

# AFlow BraTS / SAM-Med3D 单仓库权威交接（2026-08-13）

更新时间：2026-08-13 UTC

这是当前科研主线的**唯一权威入口**。旧 handoff、周报和实验说明保留作证据，
但如果描述冲突，以本文件和它绑定的结果 JSON 为准。下一位 Agent 先不要翻遍历史
目录；先读完本文，运行 doctor，再决定是否需要追溯具体实验。

## 1. 一分钟结论

项目研究 BraTS 2024 的单模态 T1ce 全肿瘤分割：`WT={1,2,3}`，排除原始标签
4。当前完成了两层工作：

1. **干净 vanilla baseline 已完成。** 在本项目明确的 972/108/270 subject-clean
   划分和官方范式下，validation best-of-five Dice 为 `0.571306`，270 例 test
   scan-macro best-of-five 为 **`0.594226`**、subject-balanced 为 `0.580970`。
   这与论文 Table 2 的 vanilla `0.601` 处于同一水平，可以说“完成了本项目协议下
   可审计的基础复现”，不能说精确复现了论文未公开的划分和 prompt 细节。
2. **residual refiner 的有效改进来自鲁棒 coarse 扰动，不是灰度 loss。** 三个
   robust-shift seed + 固定 8 个镜像 TTA + validation 冻结选择的后处理，在 108
   例 validation 得到 **`0.7004189626`**；subject-balanced 为 `0.6953719491`。
   它是 validation-selected、GT-centered oracle 的开发结果，不是独立 test，
   也不是自动全体积分割结果。

另有旧 residual-refiner checkpoint 在同一 GT-centered/N4/三点击协议上的 270 例
oracle test 为 `0.7037975898`。这个数值不是 robust `0.700419` 的 test，不能把两者
拼接成“robust 方法 test=0.7038”。270 例集合已经揭示，之后禁止再用它调参。

## 2. 任务、数据和隔离边界

| 项目 | 固定约定 |
|---|---|
| 数据集 | BraTS 2024 post-treatment GLI |
| 输入 | 单通道 T1ce（manifest 中写作 `t1c`/`t1ce`） |
| 目标 | 二分类 WT = 原始标签 `{1,2,3}`；原始标签 `4` 排除 |
| 划分 | 972 train / 108 validation / 270 internal held-out test |
| 隔离 | case 和 subject 在三个 split 间均零重叠 |
| 输入尺寸 | `128 × 128 × 128` |
| 默认二值阈值 | sigmoid > 0.5，即 raw logit > 0 |

权威 manifest 在 `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/`：

| Split | 文件 | 行数 | SHA-256 |
|---|---|---:|---|
| train | `train_t1ce_wt.jsonl` | 972 | `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65` |
| validation | `validation_t1ce_wt.jsonl` | 108 | `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb` |
| test | `sealed_test/test_cases_t1ce.jsonl` | 270 | `a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9` |

原始 NIfTI 不提交 Git。当前核验过的 HDFS 挂载位置是：

```text
/mnt/hdfs/sjy/datasets/brats2024_gli_t1ce_seg_v1
```

manifest 中保留绝对 HDFS 数据路径是有意的：这是数据位置，不是代码对外部源码的
依赖。若 worker 上 `/mnt/hdfs/sjy` 不存在，需要先恢复 HDFS 挂载。不要改写已有
manifest，否则会破坏历史实验的哈希绑定。

## 3. vanilla baseline 到底做了什么

### 3.1 官方范式预处理

1. 用 TorchIO `LabelMap` 将 `{1,2,3}` 合为 WT。
2. image/label 成对重采样到 1.5 mm 各向同性；图像线性插值、标签最近邻。
3. `ToCanonical` 统一方向。
4. `CropOrPad(mask_name="label")` 以 **GT 为中心**裁到 128³。
5. 在 patch 内 T1ce 正值体素上计算均值/标准差，随后 z-score。
6. 不做 N4；只在训练时使用三个轴向的 `RandomFlip`。

请注意第 4 点：这个“官方范式 baseline”使用标签提供空间定位，因此不是完全自动、
无标签的整幅体积推理。它适合做协议内对照，但必须标注 GT-centered/oracle geometry。

### 3.2 模型与训练/评测语义

- 基础权重：`artifacts/checkpoints/sam_med3d_turbo.pth`，SHA-256
  `899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`。
- SAM-Med3D 全参数训练，不是 LoRA/partial freeze。
- loss 是标准 DiceCE；没有灰度阈值、灰度均值、课程学习或 BraTS 2022 混训。
- 严格执行上游写死的 11 次训练交互，累加 11 个 DiceCE。
- validation/test 使用五次循环 mask 点击；汇报逐 scan 的 best-of-five 和第 5 次。
- 2 × A100-SXM4-80GB，200 epoch；validation 选中 epoch 21。

baseline checkpoint：

```text
workspace/BraTS/sammed3d_t1ce_wt/
  sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023/
  sam_model_val_official_best5.pth
```

SHA-256：`c0c85a0fb90f6301344053f97cabe8a4972bd9febd4c6ae248f3a0908598fe31`。

### 3.3 baseline 结果

| Split / 聚合 | 指标 | Dice |
|---|---|---:|
| 108 validation，scan-macro | best-of-five | `0.571306` |
| 108 validation，scan-macro | click 5 | `0.559185` |
| 270 test，scan-macro | best-of-five | **`0.594226`** |
| 270 test，scan-macro | click 5 | `0.585754` |
| 270 test，subject-balanced | best-of-five | `0.580970` |

逐病例 test report：

```text
workspace/BraTS/sammed3d_t1ce_wt/
  sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023/
  test270_epoch21_official15_upstreamliteral5_s2023_20260812/test_report.json
```

SHA-256：`7bd59843459876bd006230b24bc111cdfd726896cb3e6748b54b24c0a9659b63`。

## 4. residual refiner 的科研脉络

### 4.1 旧 refiner：证明“细化 coarse logit”有价值

旧方法把 N4-corrected T1ce 和冻结 SAM 第三次点击的 raw coarse logit 输入
SegResNet residual refiner。空间处理是 native-resolution `ToCanonical` 后
GT-centered `CropOrPad(128³)`，仍是 oracle 定位。

validation 上，epoch 58 将同次 H100 coarse `0.6359485294` 提高到
`0.6683989912`，提升 `+0.0324504618`，93/108 例改善。`0.6800` 只是预注册成功
门槛，从来不是实际成绩。

在 2026-08-13 用户明确授权并冻结协议后，旧 epoch-58 checkpoint 在已经揭示的
270 例 test 上得到：

| 聚合 | coarse | refined | 配对提升 |
|---|---:|---:|---:|
| scan-macro | `0.6666580346` | **`0.7037975898`** | `+0.0371395552` |
| subject-balanced | `0.6488642544` | `0.6866373805` | `+0.0377731261` |

244/270 例改善。结果绑定文件：

```text
workspace/BraTS/sammed3d_t1ce_wt/FINAL_TEST_RESULT_RESIDUAL_REFINER_20260813.json
```

SHA-256：`149fc9a89d5877de9efa6e52f1a8a459d7c918eb2aabf06a3872846bd20fdcdd`。
这是 oracle test 机制证据，不是自动方法，也不是 pristine external test。

### 4.2 发现真正问题：coarse 输入 train/validation 分布漂移

固定 coarse cache 的平均 Dice 显著不同：

- train frozen coarse：约 `0.829776`；
- validation frozen coarse：约 `0.636588`。

这说明旧 refiner 在训练时看到的 coarse mask 过于准确，到 validation 面临定位误差
时泛化不足。后续有效改动不是再加一个语义 loss，而是直接模拟这种误差。

### 4.3 robust-shift：有效的主改动

训练期间，对 coarse raw logit 独立做每轴最多 4 voxel 的随机平移，越界处补零；
**图像和标签不做这个额外平移**。因此模型不得只依赖 coarse mask 与 GT 的过度对齐，
而需要从图像重新判断边界。推理时不加随机平移。

三个无灰度 loss 的 seed，raw validation scan-macro Dice：

- seed 20260813：`0.6956362921`；
- seed 20260814：`0.6963512466`；
- seed 20260815：`0.6990433552`。

三模型 raw logit 等权 ensemble：`0.6979023964`；三模型各做 8 个固定轴向镜像、
逆变换后 float32 等权平均的 raw 结果：`0.6982346554`。

最后使用**同一 validation 预先冻结选择**的：

- raw logit threshold = `-1.0`；
- 删除 26-connectivity 下小于 500 voxel 的连通域；

得到 validation scan-macro **`0.700418962602465`**、subject-balanced
`0.6953719490880098`。相对 A100 replay epoch58 base `0.6695482504`，配对提高
`+0.0308707122`，96/108 例改善，case bootstrap 95% CI 为
`[+0.0259445,+0.0358042]`。

最终权威结果：

```text
workspace/BraTS/sammed3d_t1ce_wt/
  FINAL_VALIDATION_RESULT_RESIDUAL_REFINER_ROBUST_ENSEMBLE_20260813_V2.json
```

SHA-256：`74d5b3018be637bb1bd05adea0ceb99cdaeb38b5bb2af5f52b4566301df6f186`。

证据边界必须一起写：`0.700419` 只比 0.70 高 `0.000419`，最终点估计自身的
bootstrap 区间为 `[0.666711,0.731399]`；threshold 和连通域规则也在同一 108
例 validation 上选择。因此它是 validation-selected 开发成绩，不代表总体 Dice
已经统计显著超过 0.70，下一步必须冻结整套协议后去新的外部确认集。

### 4.4 三个最终 robust checkpoint

| Seed | 相对路径 | SHA-256 |
|---:|---|---|
| 20260813 | `workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_robustshift4_graymean_paired_a100_20260813/robustshift4_control/refiner_val_dice_best.pth` | `011a7b0f4a9ee193883662f03fc6d8a54cef9fd3b4c08c2871cdddbdfbe73f6a` |
| 20260814 | `workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_robustshift4_three_seed_ensemble_20260813/robustshift4_seed20260814/refiner_val_dice_best.pth` | `c94f32c04cb89f51e1b83ae157bbce1d89371aa21b4267164e1ed645fa4d4cbc` |
| 20260815 | `workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_robustshift4_three_seed_ensemble_20260813/robustshift4_seed20260815/refiner_val_dice_best.pth` | `c5d2b48db2666f097cb0d86e78c20ea95d163c07f98401714d25d755e99aceb0` |

## 5. 灰度 loss 为什么不是最终方案

gray loss 的实现是：计算 GT 肿瘤区 T1ce 平均强度，与 sigmoid(logit) 加权预测区
平均强度之间的 soft MSE。它不是“灰度阈值后处理”，也不是 baseline 原本的一部分。

严格配对结果：

| 训练 regime | gray − control |
|---|---:|
| clean continuation，weight 1 | `-0.0009623` |
| shift4 probability 1，weight 10 | `-0.0017540` |
| shift4 probability 0.5，weight 1 | `+0.0015542` |

结论是：弱 shift 条件下出现很小的条件性收益，但符号不稳定，而且该 gray arm 仍比
最佳无 gray robust-shift 模型低约 `0.010455`。因此不能宣称灰度 loss 稳定有效；
最终 `0.700419` 的完整协议中 **gray weight = 0**。

## 6. 仓库现在如何单仓库运行

### 6.1 单仓库化前后

以前的依赖关系是：

```text
AFlow source/scripts/tests
  ├─ Python import ──> ../SAM-Med3D-main/segment_anything
  ├─ SHA validation ─> ../SAM-Med3D-main/{train,inference,utils/click_method}.py
  ├─ base checkpoint > ../SAM-Med3D-main/ckpt/sam_med3d_turbo.pth
  └─ shell Python ───> ../sammed3d-venv/bin/python（若干绝对路径）
```

现在的正式关系是：

```text
AFlow/
  ├─ third_party/SAM-Med3D/             vendored 上游源码快照
  ├─ artifacts/checkpoints/             本地大权重（Git ignore）
  ├─ sammed3d/paths.py                   repo-local 默认 + 环境变量覆盖
  ├─ environment/                       最小依赖、完整 freeze、建环境说明
  ├─ scripts/aflow_env.sh                Python/路径统一解析
  ├─ data/datasets/...                   小型 manifests
  └─ workspace/BraTS/...                 cache/checkpoint/report（Git ignore）
```

默认源码路径是 `third_party/SAM-Med3D`，默认基础权重是
`artifacts/checkpoints/sam_med3d_turbo.pth`。可用环境变量覆盖：

```bash
export AFLOW_SAMMED3D_ROOT=/custom/SAM-Med3D
export AFLOW_SAMMED3D_CHECKPOINT=/custom/sam_med3d_turbo.pth
export AFLOW_PYTHON=/custom/python
```

Python 优先级：`AFLOW_PYTHON` → `AFlow/.venv/bin/python` → 临时兼容 sibling venv
→ `PATH` 中 python。正式入口不再依赖机器专属绝对路径。原 sibling
`SAM-Med3D-main` 和 `sammed3d-venv` 都保留，未删除；前者只作为只读备份。

### 6.2 vendored 第三方源码

`third_party/SAM-Med3D/` 只包含实际所需的小型源码、README 和 LICENSE，没有复制
论文 PDF、assets、演示数据、旧结果、checkpoint 或 `__pycache__`。来源目录不是 Git
仓库，所以精确 upstream commit 不可恢复；身份由逐文件 SHA 锁定：

- 来源/日期/许可证：`third_party/SAM-Med3D/SNAPSHOT_PROVENANCE.md`；
- 33 个源码文件哈希：`third_party/SAM-Med3D/SHA256SUMS`；
- 许可证：Apache License 2.0，完整文本在 vendored `LICENSE`。

upstream-literal trainer 校验的三个关键 SHA 仍与历史完全一致：

| 文件 | SHA-256 |
|---|---|
| `train.py` | `474d9765e29c0196e9cf1da2242869c2cbd6970632d9f113357d38a8094e1e3f` |
| `inference.py` | `5a31aff32805f297eb5632b4362d1aa60bde4491d168868bc750a43e332c3960` |
| `utils/click_method.py` | `0838972fb2f33de579a2652d3225f862d84dc6df113b152ea6018730c9b792d7` |

### 6.3 大文件与 Git

以下内容保存在 AFlow 磁盘中但由 `.gitignore` 排除，不会进入普通 Git commit：

- `artifacts/checkpoints/sam_med3d_turbo.pth`，约 384 MiB；
- `workspace/BraTS/` 下全部 cache、checkpoint、日志和结果；
- `data/datasets/` 下的数据/manifest 资产；
- `.venv/`。

基础权重是从原文件普通 copy 得到的（reflink 不受当前文件系统支持），原文件保留，
两份 SHA 一致。checkpoint/JSON/CSV/日志/Markdown 未做清理或重命名。AFlow 工作树
原本就有大量用户删除和 untracked 科研资产，本次没有使用 `git reset`、`git clean`
或 `checkout`，也没有把历史实验目录挪走。

## 7. 易失 `/tmp` 缓存的处理

原 `/tmp/aflow_refiner_graymean_20260813` 曾约 4.3 GiB，包含 972 train + 108
validation 的冻结 SAM click-3 float16 logits。审计时该 worker 临时目录已经不存在，
所以不能“复制原缓存”；我们没有伪造旧 index，也没有删除任何 `/tmp` 内容。

它已从持久 source cache、固定 coarse checkpoint 和 seed 20260723 在 2 × A100 上
实际重建到：

```text
workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813/{train,validation}
```

新 index 的 `source_cache.path` 使用 AFlow repo-relative 路径，不再写机器绝对路径。
实际完整重建精确得到：

| Split | 数量 | coarse DSC | index SHA-256 | logits SHA-256 |
|---|---:|---:|---|---|
| train | 972 | `0.8297756239` | `34e9322a45b6501d841e2fab1dba43ef68353e767146ca547e5e8bb280aa95bf` | `bb39991978e514ab79980657defde084cbca34ba041b9b0c324b41cd11cdaf67` |
| validation | 108 | `0.6365875326` | `72fddef3c7a8755f0712b864efbb7328e2d8ae97313ac7ca7a0306d4b612ce44` | `7d7d4e3f21ed9ef873768486672f2043d3951b8363b2f6745d4ae86092a20fde` |

完整迁移收据在缓存根目录的 `MIGRATION_RECEIPT.json`。其中 train logits
4,076,863,488 bytes，validation logits 452,984,832 bytes。

重建/核验入口：

```bash
bash scripts/rebuild_t1ce_wt_refiner_cache.sh verify
# 只有缓存确实丢失且目标目录不存在时才运行：
bash scripts/rebuild_t1ce_wt_refiner_cache.sh build
```

脚本拒绝覆盖已有 split。不要为了“重跑更干净”删除现存 cache。

## 8. 环境、doctor 与最快启动

从 AFlow 根目录执行：

```bash
cd /mlx_devbox/users/songjunyi.666/playground/AFlow
bash scripts/doctor_brats_single_repo.sh
bash scripts/run_worker_research.sh probe
```

这三条就是下一位 Agent 的最短启动序列。doctor 会核验：

- 当前 Python、torch/CUDA 与前两张卡是否均为 A100；
- `segment_anything` 是否实际从 vendored AFlow 路径 import；
- 33 个上游文件及三个 upstream-literal 关键 SHA；
- 基础权重、baseline/coarse/legacy/三个 robust checkpoint 和最终结果 JSON；
- 972/108/270 manifest 的行数和 SHA；
- manifest 样本数据路径；
- GT-crop N4 source cache 及持久 frozen-logit refiner cache。

在无 GPU 的开发机上只做代码/资产审计可临时运行：

```bash
bash scripts/doctor_brats_single_repo.sh --allow-no-gpu
```

这不是正式验收；最终正式 doctor 必须在两张 A100 可见时无豁免通过。环境建立说明和
依赖锁在 `environment/README.md`、`environment/requirements-medical.txt` 和
`environment/sammed3d-venv.freeze.txt`。

## 9. 当前代码入口和历史资产

| 用途 | 入口 |
|---|---|
| 路径解析 | `sammed3d/paths.py`、`scripts/aflow_env.sh` |
| 单仓库 doctor | `scripts/doctor_brats_single_repo.sh` |
| worker 总入口 | `scripts/run_worker_research.sh` |
| official baseline trainer | `scripts/train_sammed3d_t1ce_wt_upstream_literal_va.py` |
| baseline test evaluator | `scripts/evaluate_t1ce_wt_upstream_literal_test.py` |
| refiner frozen-logit cache | `scripts/cache_sammed3d_t1ce_wt_refiner_inputs.py` |
| refiner cache 重建/核验 | `scripts/rebuild_t1ce_wt_refiner_cache.sh` |
| residual refiner trainer | `scripts/train_sammed3d_t1ce_wt_residual_refiner.py` |
| robust gray/shift trainer | `scripts/train_t1ce_wt_residual_refiner_graymean_robust.py` |
| 三 seed ensemble/TTA | `scripts/evaluate_t1ce_wt_residual_refiner_ensemble*_validation.py` |
| postprocess calibration | `scripts/calibrate_t1ce_wt_residual_refiner_postprocess.py` |

历史训练记录会包含当时的绝对路径和历史代码 SHA，这是不可改写的 provenance，不代表
当前活动源码仍依赖那些路径。不要批量“修正”已有 JSON/日志。此前只读清理审计在
`archive/research_documents/2026-08-17_root_cleanup/CODE_CLEANUP_AUDIT_20260813.md`；因为大部分科研源码是 Git untracked，本次没有
物理删除/移动历史代码。

## 10. 下一步科研建议

优先级建议：

1. **先冻结 robust 完整协议并寻找新外部/确认集。** 这是验证 0.700419 是否泛化的
   唯一干净方式。不要在现有 270 例上再做 robust test 或选择 threshold/TTA。
2. **把 oracle 定位换成自动定位。** 当前 robust refiner 仍吃 GT-centered crop；
   真正可发表的系统需要用自动 brain-crop/coarse localization，并在 validation 上做
   完整的失配鲁棒性训练。
3. **做 official-protocol-matched brain-crop 单因素实验。** 旧自动 brain-crop 在
   旧 N4/三点击 test 协议达到 `0.625359`，有希望但不能与最新 official15 baseline
   直接因果比较。固定其余因素，只替换空间几何，在 972/108 上重新验证。
4. **灰度 loss 降级为次要/负结果。** 除非提出更清晰的强度建模假设和独立验证，
   不要继续用小于 0.002 的不稳定波动包装为主创新。

## 11. 绝对禁止的证据越界

- 禁止把 GT-centered/oracle crop 写成自动端到端分割。
- 禁止把 `0.700419 validation` 写成 test，或把旧 `0.703798 oracle test` 冒充
  robust 三 seed 方法的 test。
- 禁止用已经揭示的 270 例选择 checkpoint、loss、阈值、TTA、连通域大小、crop
  或任意后处理；它也不能再称 pristine/unseen test。
- 禁止把论文 `0.601` 说成本项目“精确复现目标”；论文未公开足够协议细节。
- 禁止把灰度 loss 写入 baseline 或最终 robust 协议；两者都不含它。
- 禁止把不同预处理、点击数、N4、定位方式的数字组成单因素 leaderboard。
- 禁止改写历史 result/freeze/report 来匹配当前路径或当前代码 SHA。
- 禁止提交 BraTS 原始数据、cache 或大 checkpoint 到普通 Git 历史。

## 12. 结果解释的推荐一句话

> 在 BraTS 2024 T1ce/WT 的 972/108/270 subject-clean 划分上，我们完成了一个
> test best-of-five Dice 0.594226 的可审计 vanilla SAM-Med3D baseline；针对
> GT-centered residual refinement 的 train/validation coarse-logit 失配，通过
> 训练时随机平移 coarse logit、三 seed ensemble、固定 TTA 和 validation-selected
> 后处理，将 108 例 oracle validation Dice 提高到 0.700419。该结果不含灰度
> loss，尚需在新的外部确认集和自动定位条件下验证。

这句话保留了真实进展，也没有越过当前证据边界。

## 13. 尚未解决的工程与证据风险

- AFlow 仍是很脏的工作树，许多科研文件是 Git untracked；磁盘上的单仓库可运行，
  但尚未形成一个可远程 clone 的干净 commit。提交前必须逐项 review，不能用
  `git clean/reset` 代替整理。
- `workspace/`、`data/datasets/` 和 `artifacts/checkpoints/` 被 Git 忽略；它们已在
  当前磁盘持久化，但没有自动的远端 artifact 备份。换机器前需要按本文 SHA 另行同步。
- vendored 来源目录不是 Git checkout，精确 upstream commit 未知；逐文件 SHA 能
  重现实有源码，但不能恢复完整上游版本史。
- 当前 Python 首选 `.venv`，但本机仍实际使用相邻旧 venv fallback。建立并验收
  AFlow `.venv` 后，才能彻底取消这个临时兼容分支。
- 管理式 GPU worker 需要 `/tmp/aflow_nvidia_535_161_08` 里的 driver compatibility
  libraries；`aflow_env.sh` 会自动发现它，但新 worker 若没有该目录，需要由平台重新
  准备对应驱动库，doctor 会 fail-closed。
- 单仓库迁移改变了若干活动源文件字节，因此历史 prereg/freeze 的旧源码 SHA 不应
  再授权新运行。历史 July rank0 preflight 已明确 fail-closed；任何重跑必须新建
  prereg，禁止改写旧绑定。
- 科研上最关键风险仍是 oracle 定位、同一 validation 上的后处理选择，以及 270 例
  test 已揭示。新的外部确认集和自动定位实验是下一阶段必需证据。

## 14. 2026-08-14 MCTS 推理效率进展（最新）

### 14.1 这部分实际做了什么

在 robust residual refiner 的 3 个无灰度 loss checkpoint 已固定后，将“选哪些模型、
选哪些反射视图、怎样融合、用什么阈值和连通域后处理”定义成强类型 workflow，使用
真正的 UCT MCTS 自动搜索。搜索协议是 subject-disjoint 的 nested search / selector /
outer 三折，只能访问 108 例 validation 的 float32 atom cache；270 例 test 在入口不可达。

初始正式三折 MCTS 得到：

| 指标 | MCTS OOF | 24-forward locked reference | 差值 |
|---|---:|---:|---:|
| scan Dice | `0.699821` | `0.700422` | `-0.000601` |
| subject Dice | `0.695037` | `0.695373` | `-0.000336` |
| 平均 member-view forwards | `7.0` | `24.0` | `-70.8%` |

subject 配对差的 95% CI 为 `[-0.001989,+0.001365]`。所以这不是 Dice 提升结果，
也没有做出统计非劣声明；它说明在当前 oracle validation 上可以删除大量冗余推理，
而精度只发生很小变化。三个 fold 的 winner 分别需要 8、12、1 次 forward，7 是 OOF
平均成本，并不是一个可直接部署的 7-forward workflow。

### 14.2 为什么能省计算

locked reference 固定执行 `3 checkpoints x 8 reflection views = 24 forwards`，但不同
成员和视图给出的 mask 高度相似。12 例 label-free 诊断中，同模型不同视图的二值 mask
平均两两 Dice 约 `0.9457`，identity 模型之间约 `0.9488`；单 seed identity 与完整
24-atom fusion 的 mask Dice 约 `0.9555`。搜索目标又显式惩罚 forward 数，因此会保留
互补信息、删除重复成员/视图。当前三个 fold winner 都是 fixed workflow；虽然已实现
3→12→24 adaptive escalation，但它没有赢得本轮选择。

### 14.3 等预算控制实验与诚实结论

随后在完全相同的 3 folds、每 fold 128 个 unique workflow、selector top-8、mandatory
anchor 协议下比较 MCTS、random 和 greedy：

| controller | OOF subject Dice | 相对 locked | 95% CI | forwards |
|---|---:|---:|---:|---:|
| MCTS | `0.695037` | `-0.000336` | `[-0.001989,+0.001365]` | `7.000` |
| random | `0.695746` | `+0.000373` | `[-0.001139,+0.001907]` | `8.000` |
| greedy | `0.695016` | `-0.000357` | `[-0.002569,+0.001715]` | `6.667` |

MCTS 相对 random 的 subject Dice 差为 `-0.000709`，相对 greedy 为 `+0.000021`，
两者 CI 均跨 0。因此目前可作为贡献的是“精度约束、可审计的三维医学分割推理
workflow 自动优化框架及其效率发现”，不能写成“MCTS 提高 Dice”或“MCTS 优于随机/
贪心”。简单控制器表现接近本身也是重要实验结论。

### 14.4 冻结工程候选与真实在线收益

根据三折 winner 的 medoid 冻结了一个工程候选：seed `20260815` 单 checkpoint、8 个
reflection views、logit mean、threshold `-0.25`、uncertainty beta `-0.5`、移除小于
250 voxel 的 26-connectivity 连通域。workflow id 为
`a48d7c60047321da9933f5f29a924ff826b1dbcf533c4fa53e4aafbda8f63400`。

真实在线实现只执行候选需要的 8 次 forward，而不是先生成全部 24 个 atom。双 A100
交叉换卡基准（每路 warmup 3，24 cases x 2 repeats）得到：

| refiner 阶段指标 | 8-forward candidate | 24-forward locked | 收益 |
|---|---:|---:|---:|
| mean latency | `0.3871 s/case` | `1.0757 s/case` | `2.779x` speedup |
| throughput | `2.5836 case/s` | `0.9297 case/s` | `2.779x` |
| model allocated after load | `18.82 MB` | `56.46 MB` | `-66.7%` |
| peak allocated | `780.06 MB` | `952.57 MB` | `-18.1%` |

在线推理与 exact atom cache 在 candidate 和 locked 各 8 例上 mask 完全一致。这里的
延迟范围包含 refiner 阶段的 H2D、forward、融合、D2H 和连通域处理，不包含上游 SAM
coarse inference 或预处理，所以只能称 refiner-stage 加速，不能称完整端到端加速。

### 14.5 代码、结果与完整性证据

| 内容 | 路径 | SHA-256 |
|---|---|---|
| 完整贡献结果 | `archive/research_documents/2026-08-17_root_cleanup/MCTS_EFFICIENCY_CONTRIBUTION_RESULT_20260813.md` | `d0fe59509ec003a50ad9ba0bce0ee85f798763a8f72dc19c6b5d75bf55db486e` |
| 方法与预注册 | `archive/research_documents/2026-08-17_root_cleanup/MCTS_EFFICIENCY_CONTRIBUTION_PREREG_20260813.md` | `8272f77393e8523206eade1e5affcf6d6f774685ef0602ede99bc0443cd75316` |
| 初始 MCTS 正式报告 | `workspace/BraTS/sammed3d_t1ce_wt/mcts_workflow_20260813_v2/final_report.json` | `b3cabb91642e8f0b711b1a9acfd2567fa305ba57850d49d3ec5247024ed22009` |
| 等预算 controller 对照 | `workspace/BraTS/sammed3d_t1ce_wt/mcts_efficiency_contribution_20260813/FINAL_CONTROLLER_COMPARISON.json` | `757e6a82fd8be59b925010a4c692b75ad9acd6c72d0e892e46a4b7cc7cf277ff` |
| 冻结部署候选 | `MCTS_DEPLOYMENT_CANDIDATE_20260813.json` | `89ae9e10992f9989ca99a86d96bb5b153eb91050b45096940ea2a871297a986b` |
| 真实在线 benchmark | `workspace/BraTS/sammed3d_t1ce_wt/mcts_efficiency_contribution_20260813/FINAL_DEPLOYMENT_BENCHMARK.json` | `ad8cffddc4382c20af67b84674a18c707909df13cfc2cb048dd3eb5b2d471cf7` |

核心实现位于 `sammed3d/t1ce_wt_mcts.py`、`sammed3d/t1ce_wt_mcts_efficiency.py`、
`scripts/search_t1ce_wt_inference_workflow_mcts.py`、
`scripts/compare_t1ce_wt_mcts_search_controllers.py`、
`scripts/benchmark_t1ce_wt_mcts_deployment.py` 和
`scripts/verify_t1ce_wt_mcts_online_parity.py`。新增后全仓测试为 `223 passed`。

## 15. 下一 Agent 的 MCTS 开始点

1. 把论文贡献表述冻结为“精度约束、可审计的推理 workflow 自动优化及效率验证”，
   不追写不存在的 Dice 增益。
2. 若继续改算法，应预注册 Pareto/hypervolume-aware MCTS 或更有效的 adaptive gating，
   并继续保留等预算 random/greedy 对照；不能只报告 MCTS 自身。
3. 在新的外部确认集上只做一次冻结候选与 24-forward reference 的确认，避免继续使用
   已揭示的 270 例选择方法。
4. 把 GT-centered oracle 输入替换为自动定位，是从机制/效率贡献走向实际系统的关键。
5. 当前可以在论文中同时报告：Dice 基本保持的 OOF 结果、`70.8%` 平均 forward 降低、
   冻结 8-forward 候选的 `2.779x` refiner-stage 在线加速，以及 MCTS 未胜简单控制器的
   消融结论。

## 16. 2026-08-14 后续优化：adaptive 负结果与固定剪枝正收益

本节晚于第 14–15 节，若有冲突以本节为准。完整结果见
`archive/research_documents/2026-08-17_root_cleanup/INFERENCE_PRUNING_RESULT_20260814.md`（SHA-256
`0f025480349ff627a956b5e836a22925e9686f25801ea219bc1bc51c45524c9f`）。

### 16.1 Adaptive router 没有证明有效

新实现按病例执行 1→8→24 分级推理，用 subject-balanced ridge 从 prediction-only
特征预测是否值得升级。三折 strict-margin OOF 为 scan `0.700208`、subject `0.695260`、
平均 `2.556` forwards，相对 locked subject 差 `-0.000113`，95% CI
`[-0.002271,+0.002112]`。

但是 gain prediction correlation 只有 `0.0806/-0.2298`。保持每 fold 完全相同 stop
数量的 10,000 次随机路由均值为 `0.695004`，学习路由只高 `+0.000256`，且 `13.74%`
随机路由达到或超过它。因此按预注册判为负结果：不能声称 adaptive early exit 有效，
主要效率收益应归因于固定 ensemble/TTA 冗余。

### 16.2 固定预算 Pareto 曲线

| forwards | workflow | scan Dice | subject Dice | 相对 locked subject |
|---:|---|---:|---:|---:|
| 1 | seed15 identity | `0.700106` | `0.695016` | `-0.000358` |
| 3 | 3 models identity | `0.699829` | `0.695029` | `-0.000344` |
| 4 | seed15 single flips | `0.701150` | `0.696179` | `+0.000806` |
| 8 | seed15 all views | **`0.701571`** | **`0.696271`** | `+0.000897` |
| 12 | 3 models single flips | `0.700626` | `0.696146` | `+0.000773` |
| 24 | locked | `0.700422` | `0.695373` | `0` |

1/3/4/8 是当前有限候选集的 subject-Dice/forward Pareto 点；12 和 24 已被更便宜点支配。
所有相对 locked 的配对 CI 均跨 0，所以 4/8-forward 只能写成“validation 点估计略升且
计算下降”，不能写成显著 Dice 提升。

### 16.3 Fixed1 真实在线结果

两张 A100-SXM4-80GB 做交叉换卡，每路 warmup 3、24 cases × 2 repeats。fixed1 与
locked 各累计 96 个 measurement：

- latency：`0.078786` vs `1.075861 s/case`；
- speedup：`13.6555x`，两个换卡方向 `13.6362x/13.6749x`；
- forwards：1 vs 24，减少 `95.83%`；
- model allocated：减少 `66.67%`；peak allocated：减少 `26.92%`；
- candidate/locked 各 8 例 online/exact atom mask 全部逐体素一致。

这是 refiner stage 加速，不包含上游 SAM coarse 或预处理。

### 16.4 新证据与下一 Agent 起点

| artifact | SHA-256 |
|---|---|
| `workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/FINAL_ADAPTIVE_ROUTER_RESULT.json` | `dbba96e593c4bcf7742edb7f1e8afc614b058c6625ec8548e418849aef74eddf` |
| `workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/FINAL_ADAPTIVE_ROUTER_CONTROL.json` | `662ca760e061d9c207abb9733b1286be16ff533bae7ebd560f8dc0445f4f1f14` |
| `workspace/BraTS/sammed3d_t1ce_wt/fixed_budget_frontier_20260814/FINAL_FIXED_BUDGET_FRONTIER.json` | `9cc01c56659f1ca38e67ef8511ecba5376abf0e1ae1d7d1bee2567fec5480de0` |
| `FIXED1_DEPLOYMENT_CANDIDATE_20260814.json` | `e162f1bc337fb83c19163a6723a6d62f136c2518caaeb7def2d04eda6651a6d4` |
| `workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/fixed1_benchmark/FINAL_FIXED1_DEPLOYMENT_BENCHMARK.json` | `672551a39bab4bb260a2dbfbd07301584301597cef255076b83d5679f83455df` |

新增后全仓测试为 `228 passed, 26 warnings`。下一步不要继续在 108 validation 调 route
feature 或固定组合；应在新外部确认集一次性比较 frozen fixed1/fixed4/fixed8/locked24。
自动定位替换 GT-centered crop 仍是更重要的系统问题。

## 17. 2026-08-14 架构科研：显式门控负结果与 safe adapter 正收益

本节晚于前文，完整证据分别见 `archive/research_documents/2026-08-17_root_cleanup/ELPGR3D_A2_RESULT_20260814.md` 和
`archive/research_documents/2026-08-17_root_cleanup/SAFE_ADAPTER_PAIR_RESULT_20260814.md`。

### 17.1 ELPGR-A2 显式 error gate 判负

在旧 residual SegResNet 中增加 coarse uncertainty、三维 coarse boundary 与
error-supervised correction gate。参数只增加 881，但两 seed scan Dice mean 从冻结
robust control `0.697697` 降到 `0.695917`（`-0.001780`），未通过预注册。
best epoch 时 gate Dice 为 `0.58–0.69`，但平均 residual scale 仅 `0.18–0.27`；稀疏
error target 把已有有效 residual 过度压制。该方案不继续搜索 auxiliary-loss 权重。

### 17.2 SAGE-3D frozen-base safe adapter 获得正向开发结果

冻结 robust seed15 base（scan/subject `0.699043/0.694567`），外挂 1,177,050 可训练参数、
零初始化的 uncertainty-conditioned 3D adapter：

| arm | scan Dice | subject Dice | 相对 base scan |
|---|---:|---:|---:|
| plain | `0.701991` | `0.697102` | `+0.002948` |
| gated | `0.702090` | `0.697140` | `+0.003047` |

plain/gated 的 subject 配对 bootstrap CI 分别为
`[+0.001085,+0.004047]`、`[+0.000877,+0.004349]`。两臂均达到预注册 adapter 成功线。
但 gated 相对 plain 只有 scan `+0.000099`，subject CI 跨 0，不能声称 gate 有额外收益；
后续优先使用更简单的 plain arm。

运行到 epoch 10 时外部会话中断，距离 patience 正常停止差一个 epoch，未生成
`TRAIN_DONE.json`；两个 best 均为 epoch 3，并已在完整 108 例上独立 reload，scan/subject
差严格为 0。必须把它称为正向 development result，而非完整正常收尾或独立 test。
仍是 GT-centered N4 oracle refiner，270 test 未访问。

下一 Agent 不要把 mask-level GRPO 直接接到两百万体素输出。若继续 RL，应先冻结监督
adapter，再让轻量策略选择候选提示点、正负类型与 STOP，并保留 DiceCE mask decoder。

## 18. 2026-08-14 seed17 全量复现与 boundary 消融

本节晚于第 17 节，完整结果见 `archive/research_documents/2026-08-17_root_cleanup/BOUNDARY_SAFE_ADAPTER_RESULT_20260814.md`。

使用完整 972/108、同一 frozen robust seed15 base，在 GPU0 复现 plain safe adapter，
GPU1 运行严格同 seed 的 Dice-only 三维 boundary auxiliary：

| arm | scan Dice | subject Dice | 相对 base scan |
|---|---:|---:|---:|
| seed17 control/plain | `0.702191` | `0.697218` | `+0.003148` |
| seed17 boundary | `0.702632` | `0.697579` | `+0.003588` |

seed16/17 plain scan mean 为 `0.702091`，subject mean `0.697160`，说明 safe adapter 正收益
已跨两个训练 seed 复现。boundary 相对 matched control 只有 scan `+0.000440`、subject
`+0.000362`，subject CI `[-0.000626,+0.001339]`；未达到预注册 `+0.001`。尽管 boundary
head Dice 达到 `0.343`，仍不能声称它对 segmentation 有额外收益，也不应按最高单点
选择 boundary arm。

两臂均正常生成 `TRAIN_DONE.json`，best checkpoint 独立 reload 完整 108 例严格一致；
全仓测试 `240 passed, 24 warnings`。下一 Agent 应冻结 plain SAGE-3D 为架构候选，并把
boundary 保留为负/不充分消融。270 test 未访问，所有结果仍是 GT-centered oracle
development validation。

## 19. 2026-08-14 DPD启发的Domain-SAGE正式负结果

在阅读本地 `article/DPD-SAM.pdf` 后，没有直接重写SAM-Med3D主干或照搬信息不完整的
CLS/CDR，而是预注册并实现了更安全的Domain-SAGE 3D：冻结robust base，只用972个
训练病例的image-only bottleneck GAP descriptor做K-means++ (`K=4`)，再用domain MLP
和zero-init low-rank FiLM调制现有SAGE adapter的bottleneck与三级decoder。

GPU0 plain、GPU1 Domain使用相同seed `20260818`、完整972/108和相同优化协议。两臂
epoch 0均严格复现base `0.699043355/0.694567241`。正式结果：

| arm | best epoch | scan Dice | subject Dice |
|---|---:|---:|---:|
| plain SAGE | 3 | **`0.701897044`** | **`0.696969673`** |
| Domain-SAGE | 1 | `0.701764545` | `0.696724255` |

Domain−plain scan/subject差为 `-0.000132499/-0.000245418`，subject 95% bootstrap CI
`[-0.000848635,+0.000352320]`，未达到预注册 `+0.001`。Domain validation accuracy从
best segmentation epoch的`0.8611`升至early-stop时的`0.9537`，但Dice下降；因此是
“域任务学会了、却没有帮助分割”的明确负消融，不补第二seed、不继续调K/rank/权重。

plain seed18是第三次正向复现；seed16/17/18 plain scan/subject mean为
`0.702026554/0.697096458`。当前推荐架构仍为plain SAGE-3D。DPD-SAM的200 epochs属于
50-epoch prompt warm-up加150-epoch diffusion的另一套流程；本轮plain/Domain已分别在
epoch11/9按patience正常停止，继续到200只会加重已观察到的过拟合。

两checkpoint完整独立reload差严格为0，全仓测试更新为`245 passed, 26 warnings`。
完整方法、domain registry、post-hoc边界和哈希见`archive/research_documents/2026-08-17_root_cleanup/DOMAIN_SAGE_RESULT_20260814.md`，
预注册见`archive/research_documents/2026-08-17_root_cleanup/DOMAIN_SAGE_PREREG_20260814.md`，统计证据为
`workspace/BraTS/sammed3d_t1ce_wt/domain_sage_pair_20260814/FINAL_COMPARISON.json`。
270 test未访问；仍是GT-centered N4 oracle development validation。下一架构创新应转向
whole-volume自动定位/自动prompt，而不是继续在GT crop上做style-domain conditioning。

## 20. 2026-08-14 全自动定位与CAPG-3D正式结果

本节晚于第19节，完整证据见 `archive/research_documents/2026-08-17_root_cleanup/CAPG3D_RESULT_20260814.md`。

旧brain-crop的`0.625359`不能视为全自动提示结果：crop来自图像，但click1来自GT前景，
后续click来自GT误差。本轮首次建立T1ce→自动粗mask/点→冻结SAM的无GT推理闭环。
从VA/HDFS重新构建的972/108 N4 brain-crop cache四个images/labels SHA均与2026-08-06
历史cache字节一致；270 test没有访问。

第一次DiceCE 1:1配对在epoch6确认向全背景坍缩，已保留为`PROTOCOL_ABORT.json`；随后
只把两臂CE权重共同降到0.1并重新预注册，完整跑满100 epoch。结果：

| output | plain scan/subject | CAPG scan/subject | CAPG−plain scan |
|---|---:|---:|---:|
| automatic locator | `0.716125/0.706215` | `0.719090/0.710459` | `+0.002965` |
| frozen SAM dense-only | `0.596964/0.600584` | `0.595425/0.600045` | `-0.001540` |
| frozen SAM point+dense（主指标） | `0.599324/0.601878` | `0.597910/0.601175` | `-0.001414` |

主指标subject差`-0.000703`，95% CI `[-0.004178,+0.002713]`，未过预注册`+0.005`。
CAPG中心点命中率仅`0.638889`，plain segmentation argmax为`0.935185`；因此中心头正式
判为无收益。真正有用的新进展是约4.70M参数、完全自动的plain locator在108 validation
达到scan `0.7161`。旧冻结SAM会把强locator降到约`0.599`，不应继续串联；当前自动候选
应直接输出plain locator mask。

这仍是同一108例上的development结果，而且CE权重修正看过开发行为；不能称test或独立
确认。下一agent不要在108例继续调中心约束/阈值；应冻结plain locator，在新外部集确认，
或在新的开发划分上训练SAM dense-prompt/decoder适应locator生成的prompt。

配对JSON为
`workspace/BraTS/sammed3d_t1ce_wt/capg3d_dice01_pair_20260814/FINAL_COMPARISON.json`，
SHA-256 `c08a68124a111b940ce8ba14e60dcb994d64ce4b159436009a3ad6b553055aef`。
本轮结束后全仓回归为`252 passed, 26 warnings`。


## 21. 2026-08-17 plain locator 新 subject-clean 确认通过

本节晚于第20节，完整证据见 `archive/research_documents/2026-08-17_root_cleanup/PLAIN_LOCATOR_CONFIRMATION_RESULT_20260817.md`（SHA-256
`e6cddbd79c6e0ef499066a4920d37813be3f51b3bdbe896d488e501240a034e9`）。由于当前
挂载没有第二套外部 BraTS cohort，本轮没有触碰已揭示的270-case test，也没有继续在旧
108 validation调中心点或阈值；而是只从历史972-case train中预先冻结新的
756 train / 108 selector / 108 confirmation三路划分，所有role间case和subject均零重叠。

旧plain checkpoint训练时已见过全部972例，因此没有拿它直接测新confirmation。按原
plain协议从头训练新checkpoint，固定raw logit `>0`、无CAPG、无SAM后处理、无TTA或
连通域搜索。H100完整跑满100 epoch，selector选中epoch99：

- selector scan/subject：`0.720419/0.704617`；
- selector automatic point hit：`0.935185`；
- 独立reload完整逐病例报告严格相等，scan差为0；
- checkpoint SHA-256：`0bb7ca52d7ac30068f752a4fc05a859f4dce42dce6ace40ba1f6b8b332889c2f`。

随后只打开一次封存confirmation，得到scan **`0.716505`**、subject **`0.726813`**、
point hit **`0.925926`**。scan/subject bootstrap 95% CI分别为
`[0.692522,0.739225]` / `[0.700117,0.751749]`；预注册的`0.68/0.67/0.85`三个门槛
全部通过。旧development scan `0.716125`与新confirmation仅差约`+0.000380`，但病例
不同、不是配对比较，正确结论是plain locator协议获得了强内部复现，不能写成Dice提高。

易失的brain-crop cache已重建到持久
`workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train`；images/labels SHA
精确复现2026-08-14的`f772...`/`bf470...`。最终confirmation JSON为
`workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/FINAL_CONFIRMATION.json`，
SHA-256 `1cec25a2a1e603b1403dbf2d430b992eabc35aeb5353e0191334ce33fbf75e14`。

证据边界：这是同一BraTS 2024来源cohort内的subject-clean protocol replication，不是
外部确认或test。confirmation现在也已揭示，之后禁止把它变成decoder/阈值调参集。
CAPG中心头和旧冻结SAM后处理的负结论不变，当前自动候选仍是direct plain locator。
若继续SAM dense-prompt/decoder适配，只能在train/selector或新冻结inner split上开发，
最终泛化仍需真正外部数据确认。本轮结束后全仓回归为`254 passed, 24 warnings`。

## 22. 2026-08-17 plain locator 在已揭示 270 test 的一次性补充测评

用户随后明确授权在270-case test上测评当前改进。完整证据见
`archive/research_documents/2026-08-17_root_cleanup/PLAIN_LOCATOR_TEST270_RESULT_20260817.md`。本轮没有拿test挑模型：在推理前已冻结第21节
由108-case selector选出的epoch99 checkpoint（SHA-256
`0bb7ca52d7ac30068f752a4fc05a859f4dce42dce6ace40ba1f6b8b332889c2f`）、raw logit
`>0`、无TTA、无后处理、无CAPG、无SAM。test前冻结收据SHA-256为
`5388d79bbe6882825770132bfd07759feca3edc6ed3b0910d59336338df4e2db`。

完全复现训练的N4 shrink-2、image-derived brain crop、`128^3`预处理后，270 scans / 119
subjects的一次性结果为：

| metric | result | bootstrap 95% CI |
|---|---:|---:|
| scan-macro Dice | **`0.725820`** | `[0.706190,0.744411]` |
| subject-balanced Dice | **`0.709413`** | `[0.680995,0.735549]` |
| automatic point hit | **`0.900000`** | — |

test与该checkpoint的train/selector角色在case和subject上均零重叠。最终JSON为
`workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/FINAL_TEST270.json`，
SHA-256 `43b88933337e0325caaf97b6add8fbaf64a60b2574d75acaaedc9dceaea7e78b`。

证据边界必须保留：270 test早已被本项目其他arm揭示，因此这是“已揭示内部test上的补充
测评”，不是新的盲测或外部确认。test scan相对第21节confirmation数值高`+0.009315`，
但cohort不同且不配对，不能写成改进量。selector/confirmation/test scan依次为
`0.720419/0.716505/0.725820`，支持plain locator在同源cohort内稳定复现；当前最佳自动
checkpoint仍是epoch99 direct plain locator。此后禁止用270 test继续挑decoder、阈值、
TTA或后处理；SAM decoder适配只能在train/selector或新inner split开发，并需外部数据最终确认。

benchmark归档与突破机制已分类到
`archive/research_records/2026-08-17_plain_locator_test270/`。最强的直接机制证据是
旧development上locator直接mask `0.716125`，经过旧冻结SAM point+dense反而降到
`0.599324`；当前系统通过直接输出dense locator避开了该prompt/decoder分布失配。
历史匹配实验还证明image-only brain-crop相对whole-resize有`+0.054250`受控增益。
CE weight 0.1避免全背景坍缩，但小模型、sampling和augmentation没有逐项消融，禁止把
相对旧brain-crop的`+0.100461`全部归因给任何单一模块。

## 23. 2026-08-17 intensity-aware residual 在新 inner split 判负

用户提出灰度阈值后，没有在已揭示 selector/confirmation/test 上搜索 hard threshold；
本轮预注册了更保守的学习式替代：只在 plain locator 的不确定区，用归一化 T1ce、局部
z-contrast 和梯度幅值修正 logit。新的 648 fit / 108 inner-tune subject-clean 划分中，
三臂固定为 plain、参数匹配的 zero-intensity residual control、intensity residual。

严格 checkpoint 重载结果为：plain `0.728322/0.713027`，control
`0.729545/0.717166`，intensity `0.726907/0.713835`（scan/subject）。intensity−control
为 `-0.002638/-0.003331`，95% CI 分别为
`[-0.008138,+0.003299]` / `[-0.011072,+0.005886]`。更明显的是 point hit 从 control
`0.888889` 降到 `0.518519`；5 个预注册门槛失败 3 个。三臂重载最大差均为 `0.0`。

因此 `all_inner_gates_passed=false`，outer selector 未授权且未读取；confirmation 和
270-case test 同样未访问。当前 benchmark checkpoint 仍是第21–22节的 direct plain
locator epoch99。不要在这 108 inner cases 上继续搜灰度 cut-off、kernel 或 residual
scale。完整分析见 `archive/research_documents/2026-08-17_root_cleanup/INTENSITY_RESIDUAL_RESULT_20260817.md`；决策收据为
`workspace/BraTS/sammed3d_t1ce_wt/intensity_residual_inner_20260817/seed20260821/INNER_DECISION.json`。
本轮结束后全仓回归为 `259 passed, 24 warnings`。

## 24. 2026-08-18 plain locator 长时程通过 fresh-inner 与 outer selector

本节晚于第23节。用户询问100 epoch是否过短后，没有直接在已揭示selector上延长训练，
而是只从2026-08-17的648 fit cases中重新冻结540 fit / 108 fresh-tune subject-clean
inner split；昨天已揭示的108 intensity inner-tune完全排除，role间case/subject零重叠。
预注册仅允许一条时程：epoch1–100固定`2e-4`，epoch101–200 cosine降至`1e-5`；模型、
loss、sampling、augmentation、batch、raw logit阈值均不变。

完整200轮inner轨迹的pre100最佳为epoch95，scan/subject/point hit
`0.723604/0.719629/0.935185`；post100最佳为epoch168，达到
`0.746335/0.744497/0.925926`。scan/subject差`+0.022731/+0.024869`，subject 95% CI
`[+0.014792,+0.038213]`；五项inner gate全通过，两个checkpoint独立完整重载差为0。
因此可以回答：对这条新鲜轨迹，100 epoch确实截断了预注册的late-cosine收益；但当前
设计不能把收益拆成纯epoch效应与纯scheduler效应。

通过后锁定epoch168，使用原756 train cases / 346 subjects、seed `20260819`从头训练，
训练器没有validation DataLoader，只保存唯一fixed-epoch checkpoint。随后按预访问收据
在current outer selector评估一次：

| metric | epoch99 baseline | epoch168 candidate | 差值 |
|---|---:|---:|---:|
| scan Dice | `0.720419292` | **`0.731007992`** | **`+0.010588700`** |
| subject Dice | `0.704616933` | **`0.714314407`** | **`+0.009697474`** |
| point hit | `0.935185` | `0.925926` | `-0.009259` |

三项outer主门槛全部通过，epoch168升级为新的development candidate。outer描述性subject
CI为`[-0.001312,+0.020934]`，跨0，因此不能宣称统计显著。108 scans中80提高、26下降、
2持平，45 subjects中33提高；point hit净少1例，说明Dice收益并非中心点投机。直接支持
的机制是同一4.70M参数dense locator获得了更充分的低学习率收敛；推理结构、参数量和
成本不变。灰度残差、CAPG中心头和旧冻结SAM后处理的负结论不变。

新的development checkpoint为
`workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth`，
SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`。
完整归档为`archive/research_records/2026-08-18_plain_long_schedule/RESULT.md`，manifest已
逐项校验。108 confirmation与270 test均未访问；旧epoch99仍是唯一有一次270 test结果
的checkpoint，新epoch168只能在新的外部cohort上继续确认。全仓回归为
`270 passed, 24 warnings`。

## 25. 2026-08-18 epoch168 固定checkpoint的一次性test270结果

本节晚于第24节。用户明确指出最终仍需在固定270 test上测评；因此在fresh-inner已经选定
epoch168、outer三项门槛通过、全756例重训checkpoint已固定后，开放一次test。test前
冻结checkpoint、outer decision、manifest、raw logit `>0`、无TTA/后处理/SAM和唯一
输出路径。没有任何test可调参数，本结果不得触发再选epoch或阈值。

| reused internal test270 | epoch99 | epoch168 | 配对差 | 配对95% CI |
|---|---:|---:|---:|---:|
| scan Dice | `0.725820068` | **`0.741203949`** | **`+0.015383882`** | `[+0.009186,+0.021574]` |
| subject Dice | `0.709412503` | **`0.723179486`** | **`+0.013766983`** | `[+0.004123,+0.022935]` |
| point hit | `0.900000` | **`0.929630`** | **`+0.029630`** | — |

paired bootstrap 10,000次、seed`20260827`。270 scans中205提高、64下降、1持平；119
subjects中87提高、32下降。point transition为231 hit→hit、12 hit→miss、20 miss→hit、
7 miss→miss。scan和subject配对CI下界均大于0。

因此epoch168是当前项目在这套复用internal test上的新历史最高点估计，且相对epoch99的
配对收益稳定。该结论支持late-cosine长时程从fresh inner、outer selector迁移到test，
但不能把组合收益拆成纯epoch或纯scheduler效应。270此前已被多个arm揭示，所以仍不能
写成pristine blind test、外部确认或公开BraTS SOTA。

最终checkpoint：
`workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth`，
SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`。
test前冻结收据SHA-256为`30251c67669c03d4eaebcaeb445dac1d45c193392717f26a0147e8e97e79d750`；
最终报告为同目录`FINAL_TEST270_EPOCH168.json`（SHA-256
`91433c4b6b88b233d827e41b0d038f459c25ef41dec224c84e154e90f370bb8c`）；配对报告为
`TEST270_EPOCH168_VS_EPOCH99.json`（SHA-256
`62792337648b39d6080c88c9c67be915741a0c8176b1a6353c9833577d7e7a9b`）。完整归档见
`archive/research_records/2026-08-18_plain_long_schedule/RESULT.md`；全仓回归为
`273 passed, 24 warnings`。

## 26. 2026-08-18 新 1M 会话入口：160³ 分辨率单因素已授权、尚未开跑

用户已授权下一步做 plain locator 的 `128³ -> 160³` 严格单因素实验。当前会话只完成了
可行性审计，**没有创建新切分、没有改代码、没有构建160 cache、没有启动训练，也没有
再次访问outer selector、confirmation或270-case test**。新会话不要误把本节当成已完成
结果。

实验目的：验证更高的whole-brain image-only brain-crop网格能否在不改变模型结构、loss、
optimizer、sampling、augmentation、阈值或后处理的情况下，提高direct plain locator。
160 cache必须从原始NIfTI按原链路重新生成：canonical RAS -> N4 shrink-2 -> image nonzero
bounding-box crop -> trilinear image / nearest-exact label resize到`160³`；禁止把现有`128³`
cache二次插值到160。CAPG中心头、灰度residual和旧冻结SAM后处理均保持判负，不混入本轮。

建议冻结协议如下：

- 数据母集只用
  `data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/train_cases.jsonl`
  的756 cases / 346 subjects（SHA-256
  `a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16`）；重新以新seed
  冻结648 fit / 108 tune，case和subject均零重叠。必须明确写成“同一历史train cohort内
  新重采样的subject-clean开发划分”，不是外部或pristine validation。
- 匹配跑`128³` control与`160³` candidate；两臂同一split、training seed、4,700,897参数
  plain SegResNet、batch 8、DiceCE `ce_weight=0.1`、AdamW `lr=2e-4`、
  `weight_decay=1e-5`、`max_shift=8`、raw logit `>0`，无TTA/后处理/SAM。
- 为避免再用tune挑epoch造成额外自由度，primary比较固定epoch168；两臂使用相同的
  100-epoch constant + late-cosine schedule定义。建议门槛在训练前冻结为：scan和subject
  Dice各至少`+0.005`、paired subject bootstrap 95% CI下界不小于0、point hit下降不超过
  `0.02`、checkpoint完整reload差为0。未过门槛则160判负，不继续调尺度或阈值。
- 开发阶段禁止读取已揭示的selector、confirmation和test270。只有inner预注册门槛通过后，
  才另行决定是否全756重训和做一次固定checkpoint评测；test结果不得回流选择分辨率。

工程审计结果：H100 80GB在检查时空闲约81,000MiB；128³ batch8历史peak约21.5GiB，
按体素数估算160³约为1.953倍，batch8预计可行，但必须先做forward/backward smoke并记录
真实peak。现有
`scripts/build_t1ce_wt_braincrop_cache.py`、`scripts/train_t1ce_wt_capg3d.py`、
`scripts/train_t1ce_wt_plain_long_schedule.py`和`sammed3d/t1ce_wt_cached.py`均写死128，且历史
结果绑定其SHA；不要改写这些历史入口，应新增variable-resolution builder/dataset/trainer及
对应测试。

当前磁盘是实际阻塞点：文件系统仅余约`9.6 GiB`，756例float32 image + uint8 label的
true-160 cache约需`14.4 GiB`，还要留训练输出和安全余量。审计到可重建的派生cache包括
`workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32`约21GiB、旧
`t1ce_wt_gtcrop_n4_128`约11GiB和`t1ce_wt_n4_128`约11GiB；本会话没有删除任何内容。
新会话应先核对归档/重建入口并让用户确认清理一个足够大的派生cache，不能静默删除。

Codex会话本身的诊断：当前线程创建于`2026-08-17T02:38:45Z`，运行时仍锁定
`model_context_window=258400`；`/home/tiger/.codex/config.toml`在当天
`07:48:19Z`之后已配置`gpt-5.6-sol`、`model_context_window=1000000`、
`model_auto_compact_token_limit=900000`。因此应在退出/重载Codex后创建或确认一个实际报告
约1M窗口的新线程，再从本节继续。
