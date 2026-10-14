# 2026-08-17 direct plain locator / test270 科研归档

## 归档结论

- 当前项目内同一270-case internal test历史最高：scan-macro WT Dice
  **`0.725820`**，subject-balanced **`0.709413`**。
- 当前最佳自动模型：selector预先选出的epoch99 direct plain locator，checkpoint
  SHA-256 `0bb7ca52d7ac30068f752a4fc05a859f4dce42dce6ace40ba1f6b8b332889c2f`。
- selector / untouched internal confirmation / test scan分别为
  `0.720419/0.716505/0.725820`。
- 证据性质：已揭示内部test上的补充测评；不是pristine blind test、外部确认或公开
  BraTS SOTA。

本目录是逻辑归档索引。大型checkpoint、cache和逐病例JSON保留在原位，不复制、不
移动，以免破坏训练收据、代码和历史文档中的路径绑定。核心不可变文件的校验值见
[`MANIFEST.sha256`](MANIFEST.sha256)。

## A. 权威叙述与benchmark解释

| 作用 | 活跃文件 |
|---|---|
| test270完整结果、历史benchmark表、突破机制证据分级 | [`PLAIN_LOCATOR_TEST270_RESULT_20260817.md`](../../research_documents/2026-08-17_root_cleanup/PLAIN_LOCATOR_TEST270_RESULT_20260817.md) |
| 新subject-clean训练与confirmation结果 | [`PLAIN_LOCATOR_CONFIRMATION_RESULT_20260817.md`](../../research_documents/2026-08-17_root_cleanup/PLAIN_LOCATOR_CONFIRMATION_RESULT_20260817.md) |
| 当前精简结果账本 | [`AFLOW_BRATS_RESULTS.md`](../../../AFLOW_BRATS_RESULTS.md) |
| 权威handoff第21–22节 | [`AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md`](../../../AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md) |

## B. 协议冻结与数据角色

| 类别 | 文件/目录 | 约束 |
|---|---|---|
| confirmation预注册 | `PLAIN_LOCATOR_CONFIRMATION_PREREG_20260817.md` | 固定模型、loss、selector和confirmation门槛 |
| 756/108/108角色划分 | `data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/` | role间case/subject零重叠 |
| test前冻结 | `workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/TEST270_FREEZE_20260817.json` | epoch99、logit>0、无TTA/后处理/SAM |
| 固定test manifest | `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl` | 270 scans / 119 subjects |

## C. 模型、训练和选择

| 类别 | 文件 | 关键信息 |
|---|---|---|
| checkpoint | `workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/plain_seed20260819/locator_val_dice_best.pth` | epoch99；4,700,897参数 |
| 训练收据 | 同目录 `TRAIN_DONE.json` | 756 train；CE weight 0.1；100 epochs；H100 |
| selector独立reload | `workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/SELECTOR_RELOAD_VALIDATION.json` | 完整逐病例结果严格相等 |

## D. 缓存与评测输出

| 类别 | 文件/目录 | 关键信息 |
|---|---|---|
| 训练/confirmation cache | `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/` | N4；image-only brain crop；972 cases |
| test cache | `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270/` | 270/270；4-case重算通过 |
| confirmation输出 | `workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/FINAL_CONFIRMATION.json` | scan `0.716505` |
| test启动收据 | 同目录 `FINAL_TEST270.json.started.json` | 推理前固定checkpoint和threshold |
| test最终输出 | 同目录 `FINAL_TEST270.json` | 270-case逐病例记录及bootstrap |

## E. 代码入口

| 作用 | 文件 |
|---|---|
| locator/CAPG模型 | `sammed3d/t1ce_wt_capg3d.py` |
| role-aware训练 | `scripts/train_t1ce_wt_capg3d.py` |
| confirmation评测 | `scripts/evaluate_t1ce_wt_plain_locator_confirmation.py` |
| 固定test270评测 | `scripts/evaluate_t1ce_wt_plain_locator_test270.py` |
| brain-crop cache | `scripts/build_t1ce_wt_braincrop_cache.py` |

## F. 正负证据归类

### 已直接证明

1. direct locator在旧108 development为`0.716125`；接旧冻结SAM dense-only和
   point+dense后降到`0.596964/0.599324`。因此当前decoder/prompt接口会破坏强locator。
2. 旧匹配实验中brain-crop相对whole-resize从`0.571109`提高到`0.625359`，配对
   `+0.054250`，95% CI `[+0.0455,+0.0629]`。
3. CE weight 1.0的首次locator训练接近全背景；共同降到0.1后正常训练。
4. 新subject-clean selector、confirmation和test均稳定在约`0.72` scan Dice。

### 正式负结果

1. CAPG不是突破来源：locator只比plain高`+0.002965` development scan，中心点命中
   `0.638889`，且primary frozen-SAM point+dense比plain更差。
2. 旧冻结SAM不是有益后处理；不得用test重新挑dense scale、threshold或组件。

### 尚未单因素证明

- 4.70M SegResNet相对100.5M SAM-Med3D的独立架构贡献；
- lesion-volume-aware sampling、flip/shift/intensity/noise各自的Dice贡献；
- 当前locator中brain-crop相对whole-resize的独立增益。

因此不能把相对旧brain-crop test的`+0.100461`全部归因给单一模块。可辩护的总解释是：
有效的image-only空间聚焦奠定输入基础，任务对齐的dense locator学出了强mask，而直接
输出避开了已实测失败的冻结SAM prompt/decoder接口；其余训练设计是合理但未逐项消融的
共同因素。

## 后续使用限制

- 270 test和108 confirmation均不得用于选择decoder、checkpoint、threshold、TTA、
  connected components或后处理。
- locator-conditioned SAM decoder只能在train/selector或新冻结inner split开发。
- 任何外部泛化或公开SOTA结论都需要新的外部cohort和匹配的BraTS评测定义。

