# 科研代码迁移指南（2026-10-14）

这份包用于在新电脑上接续研究、查阅实验过程并重建环境。包内不包含原始医学影像、预处理大缓存或模型权重；因此解包后可以阅读代码、协议和结果，**不能直接推理或逐位复现旧结果**。以下科研路径均以解包后的 `AFlow/` 为根；`article/` 与 `AFlow/` 并列。旧机器绝对路径需要重新映射。

从GitHub克隆后核对：

```bash
git clone https://github.com/songjunyi35-ctrl/Medical_research.git
cd Medical_research
python3 verify_migration_bundle.py
```

`MANIFEST.sha256.tsv`逐文件记录本次快照的路径、大小和SHA-256；这是当前科研工作树快照，不含上游AFlow Git历史、虚拟环境或`merlin-demo/`。根README列出最新结果；`AFlow/README.md`为科研入口。

原仓 `scripts/verify_aflow_brats_doc_layout.py` 会检查研究文档中指向 checkpoint 的链接；本包故意省略权重，因此在解包目录运行时会报告 `checkpoint_epoch193.pth` 链接缺失。这是已知的迁移包边界，原仓同一检查通过；迁移包的文件完整性以 `verify_migration_bundle.py` 为准。

## 先看研究状态

- 研究从单 T1ce MRI → 预处理 → SegResNet → WT展开，最新已完成完整nnU-Net残差编码器参考与有限融合；目标仍为WT（二值整体肿瘤，标签 1/2/3，排除切除腔 4）。入口 `README.md`、`AFLOW_BRATS_HANDOFF_CURRENT.md`、`AFLOW_BRATS_METHOD_CATALOG.md`、`AFLOW_BRATS_RESULTS.md`。`article/MedTAWS_GRPO/` 是另一条论文与历史方法材料，不能把其交互式结果与当前全自动主线直接比较。
- 固定原始分割基线：原 972 例纯 SegResNet，第 193 轮在复用内部 270/119 上 Scan Dice 0.759179；完整证据在 `archive/research_records/2026-09-17_segresnet_plain972_e193_test270/`。2026-09-29 多尺度 166 轮一次内部 270 测试在原生栅格为 0.771386，但验证提升门未过、历史上已接触测试队列；详情见 `archive/research_records/2026-09-29_segresnet_multiscale_e166_test270/RESULT.md`。这些都不是新盲测或外部确认，跨栅格分数不能直接相减。
- 历史实验、失败记录、冻结方案、访问收据保留在 `archive/`。旧 SAM 权重已在 2026-09-18 清理，旧实验不能因文档中仍有入口而自动重跑。

## 2026-10-14新增参考与0.8以上结果

完整nnU-Net残差编码器L参考已完成预算550轮，108验证选择525轮（Scan 0.784672）。原生1mm内部270/119单模型Scan 0.811283、Subject 0.795536、Point 265/270；最佳固定融合Scan 0.812965、Subject 0.797512、Point 266/270。所有新成绩均为复用内部队列的test-tuned探索，不是独立确认。完整报告与验证曲线在`AFlow/archive/research_records/2026-10-13_nnunet_resenc_reference/`，代码在`AFlow/scripts/nnunet_resenc_20261013/`。

新单模型权重原位置`workspace/BraTS/nnunet_t1ce_wt/resenc_20261013/model/fold_0/checkpoint_best.pth`，SHA-256 `31f979ac3574b896e3bb85b6c5a410688b82de07a5ac5a26f73b7e1c7b13806e`，本Git仓库不包含该权重。融合还需旧native173、旧组合166权重及各自预处理。保持原冻结源码不变；迁移运行前需单独记录路径映射。原现场环境为Python 3.11、torch 2.7.1+cu126、numpy 2.4.6；新增依赖见`AFlow/environment/requirements-nnunet-resenc.txt`。

## 新电脑环境

在 `AFlow/` 根目录运行（Linux、合适的 NVIDIA 驱动/CUDA GPU 是历史 GPU 训练路径的条件）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r environment/requirements-medical.txt
bash scripts/doctor_brats_single_repo.sh
```

上述命令来自 `environment/README.md`；完整旧环境快照见 `environment/sammed3d-venv.freeze.txt`。原机器旁置 `sammed3d-venv/` 是虚拟环境，不需迁移。`AFLOW_PYTHON=/新环境/bin/python` 可覆盖解释器。系统 CUDA/驱动与 PyTorch wheel 必须匹配；不应假设 CUDA 12.6 wheel 可在所有机器直接运行。`doctor` 的部分检查要求本包故意省略的历史权重或缓存，因此报告缺失并不代表打包损坏；以其代码/依赖检查为参考，不能要求全绿。

## 数据怎样取得与对齐

- 主数据是 BraTS 2024 post-treatment GLI 的本地 Kaggle version 2 镜像，历史记录为 1350 scans/613 subjects。官方 Synapse corrected-v2 训练包实体 `syn60086071` version 2；需用户自行登录并同意下载条款。**现有审计未证明镜像与官方 archive 逐文件相同**，不能改写为官方源已核验。来源、实体、access requirement、待做的 MD5 与逐文件 SHA 检查见 `archive/reviews/2026-09-05_brats2024_source_provenance_recheck/RESULT.md`，现有验证器为 `scripts/verify_brats2024_source_provenance.py`。
- 固定分割角色清单在 `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/`：`train_t1ce_wt.jsonl` 为 972 scans/444 subjects，`validation_t1ce_wt.jsonl` 为 108/50，`sealed_test/test_cases_t1ce.jsonl` 为 270/119。原 SHA-256 依次为 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`、`f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb`、`a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9`。计数、患者隔离与前两个哈希由 `scripts/sammed3d_full972_protocol.py` 检查；三份哈希另见基线 `DATA_AUDIT.json`。保留原清单及 SHA，不要重新随机划分。
- 清单里的 `image_path`、`label_path` 是旧 `/mnt/hdfs/sjy/datasets/...` 绝对路径。下载/取得同源影像后，在新机器按 case ID 映射到本地路径；修改清单会改变冻结 SHA，故应生成明确标记的新本地映射副本，并记录新旧 SHA，不可声称它仍是原始冻结清单。现有构建脚本 `scripts/build_segresnet_plain972_no_n4_cache.py` 会从清单绝对路径读 NIfTI，且假设历史 N4 对照缓存存在；不能在空白机器直接运行。历史缓存本身不在包内，需依据对应实验协议重建并核对缓存索引的病例顺序、图像/标签哈希和几何；重建后数值环境差异需单独审计。
- 公开 334 例预训练数据另来自 OpenNeuro `ds007045` v1.0.1，官方版本入口、334 对对象的 URL/SHA 及排除 3 例的规则在 `archive/reviews/2026-09-06_openneuro_ds007045_source_audit/RESULT.md` 和 `SOURCE_INVENTORY.json`。其他外部队列请从相应 `archive/reviews/*source_audit/RESULT.md` 查来源与限制，不把它们误作 972/108/270 主划分。

## 权重如何恢复

- **固定纯 SegResNet 基线 e193**从随机初始化训练，无公开基础权重依赖；入口 `scripts/train_segresnet_plain972_baseline.py`，训练配方与冻结边界在 `archive/research_records/2026-09-17_segresnet_plain972_baseline/IDEA.md`。原 `archive/research_records/2026-09-17_segresnet_plain972_e193_test270/checkpoint_epoch193.pth` 的 SHA-256 为 `157fc5af363fc42b2c993df07639693ed8d9f193b8cadd11810a39f62c0f54a5`。它**没有公开下载地址**；取得相同数据后可按记录重训方法，但随机性与硬件差异使重训不等于找回原权重或原分数。其他自训练 checkpoint 同理，去对应 `archive/research_records/<实验>/RESULT.md`、`FREEZE.json`、`SELECTION_RECEIPT.json` 查训练来源和 SHA。若权重未携带、旧盘不可用，旧 checkpoint 无法从结果文件逆向恢复。
- **最近多尺度探索 e166** 的原权重在 `archive/research_records/2026-09-29_segresnet_multiscale_e166_test270/checkpoint_epoch166.pth`，SHA-256 为 `0645c5e781571ceff2e6d26c62ca52e12e985cf2c1b097f07d7b8a0bc7b48ce2`。它从 OpenNeuro 334 例所得的本项目预训练权重开始，经原 972 例的 50 轮阶段与续训至 200 轮后由 108 验证选择 e166；链条见该目录及 `archive/research_records/2026-09-29_segresnet_native_multiscale_resume50/RESULT.md`。该候选未通过原验证提升门，不能自动替代 e193 基线。完整重训还需恢复前置预训练数据和阶段，不能保证得到同一 SHA 或结果。
- **历史 SAM-Med3D 基础权重** `sam_med3d_turbo.pth` 的上游链接在 `third_party/SAM-Med3D/readme.md` 的 checkpoint 段，Hugging Face 页面为 <https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth>。旧期待 SHA-256 `899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`，默认位置 `artifacts/checkpoints/sam_med3d_turbo.pth`，也可设 `AFLOW_SAMMED3D_CHECKPOINT=/新路径/sam_med3d_turbo.pth`；自行取得后用 `sha256sum` 核对。当前 SegResNet 纯基线不依赖它；恢复 SAM 历史方法时还须逐实验确认源码、数据、训练权重与协议。
- **公开 334 例 SegResNet 预训练权重**由本项目在 OpenNeuro 数据上训练 150 轮后生成，随后用原 972 例微调；它并不是可从官方直接下载的公开模型。详见 `archive/research_records/2026-09-21_segresnet_external_pretrain/IDEA.md` 和 `RESULT.md`。缺权重时须重建数据并按冻结方案重训，不能把 SAM-Med3D 权重替代。

## 解包后最低限度核对

1. 按开头命令校验压缩包和包内文件；确认 `AFlow/scripts/`、`AFlow/sammed3d/`、`AFlow/data/datasets/`、`AFlow/archive/` 和 `article/` 存在。
2. 进入 `AFlow/` 按上节建环境；打开 `README.md`、当前交接与具体实验 `RESULT.md` 确认要接续哪一条线。
3. 取得所需原始数据并解析绝对路径，核验角色清单、病例/患者隔离和来源限制；仅在有明确研究方案时重建缓存与训练。
4. 要做推理，先取得相应实验的**模型权重**并验证 SHA，以及匹配的预处理与输入栅格。仅有本代码包和结果 JSON 无法直接生成旧模型预测。

迁移不会自动批准新的测试集访问、实验扩组或论文结论；历史报告中的权限、负结果和数据来源限制需要继续保留。
