# AFlow 当前科研计划

最后更新：2026-08-18 UTC

## 当前状态

Epoch168 plain locator 实验已经完成。它在固定 270-case internal test 上的全自动
scan/subject Dice 为 `0.741204/0.723179`；该结果不得用于后续选择。

下一项实验已获得原则授权，但**尚未开始**：对同一个 direct plain locator 进行严格的
`128^3` 与 `160^3` 输入分辨率单因素比较。目前尚未为该实验创建新 split、预注册、
cache、代码或 smoke，也没有启动训练。

## 当前阻塞项

上次审计发现磁盘仅剩约 `9.6 GiB`，而 756 例 true-160 cache 预计约需 `14.4 GiB`，
且这还不包含 checkpoint 和安全余量。开始前必须重新检查磁盘空间。

可能可以重建的派生缓存包括：

- `workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32`（约 21 GiB）
- `workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128`（约 11 GiB）
- `workspace/BraTS/cache/t1ce_wt_n4_128`（约 11 GiB）

不得静默删除任何一个缓存。必须先核验重建命令和产物依赖；如果空间仍不足，再向用户
申请删除某个准确目标。

## 已授权的下一项实验：分辨率单因素

### 科研问题

在不改变模型架构、loss、optimizer、sampling、augmentation、推理阈值和后处理的前提下，
将 image-only 全脑裁剪网格从 `128^3` 提高到 `160^3`，能否改善 direct plain locator
的 Dice？

### 数据隔离

- 母 cohort 只能使用下列 manifest 中的 756 cases / 346 subjects：
  `data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/train_cases.jsonl`。
- 母 manifest SHA-256：
  `a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16`。
- 编码或训练前，使用新 seed 预注册一个 648 fit / 108 tune 的 subject-clean 划分，
  case overlap 和 subject overlap 必须均为 0。
- 这是对同一个历史训练 cohort 的新重采样，不是外部或纯净 validation 集。
- 分辨率开发期间，现有 selector、confirmation 和 270-case test 必须不可访问。

### 匹配实验臂

运行一个 `128^3` control 和一个 `160^3` candidate，并严格保持以下条件相同：

- split 与训练 seed；
- 4,700,897 参数的 plain SegResNet；
- batch size 8、subject/lesion sampling 和 augmentation；
- DiceCE，`ce_weight=0.1`；
- AdamW，`lr=2e-4`、`weight_decay=1e-5`；
- `max_shift=8`；
- epoch 1-100 固定 `2e-4`，之后使用相同的 late-cosine 定义；
- primary 固定在 epoch168 读取；
- raw logit threshold `>0`；
- 不使用 SAM、CAPG、TTA、connected components 或其他后处理。

不得把 gray residual、CAPG、safe adapter 或 frozen-SAM decoding 混入本实验。

### True-160 预处理

必须直接从原始 NIfTI 使用现有语义链构建 160 cache：

`canonical RAS -> N4 shrink-2 -> image nonzero bounding-box crop -> trilinear image / nearest-exact label resize to 160^3`

绝不能通过插值现有 128 cache 来生成 160 cache。

历史 128-only builder、dataset 和 trainer 都与哈希绑定，不得原地修改。应新增带版本号的
variable-resolution builder、dataset、trainer、launcher 和测试。

### 预注册成功门槛

两条正式实验臂启动前必须冻结以下门槛。160 candidate 相对匹配的 128 control 必须全部满足：

1. scan Dice 差值至少为 `+0.005`；
2. subject-balanced Dice 差值至少为 `+0.005`；
3. paired subject bootstrap 95% CI 下界不小于 `0`；
4. automatic point-hit 下降不超过 `-0.02`；
5. 完整 checkpoint 重载后的逐病例 Dice 最大差等于 `0`。

只要有一项失败，就把 160 记录为负结果。不得在同一个 tune 集上继续调整其他分辨率、
阈值、loss weight、epoch 或后处理。

## 执行顺序

1. 运行 `df -h`，盘点三个候选缓存；在不删除任何未获批准内容的前提下解决磁盘阻塞。
2. 在实现正式实验前，写入并哈希冻结预注册和新的 subject-clean split。
3. 新增 variable-resolution cache、dataset、trainer、launcher、analysis 以及隔离/重载测试，
   不修改历史入口。
4. 先完成一个真实病例的 128/160 预处理一致性检查，再构建小规模 cache smoke。
5. 在现有 2 张 A100 80 GB 上完成前向/反向 smoke，并记录真实峰值显存。不能仅凭估算
   假定 batch 8 一定可行。
6. 两条实验臂都通过 smoke 后，一张 GPU 运行一个实验臂，并发执行匹配的 128 和 160，
   日志与输出目录必须分离。
7. 在全部 108 个 tune cases 上重载两个固定 epoch168 checkpoint，计算 scan/subject Dice、
   point hit、paired bootstrap 和冻结门槛决策。
8. 更新 `plan.md`；若稳定结论改变则更新 `project.md`；同时更新结果总账、周报、权威交接，
   并建立新的自包含归档记录。
9. 只有 inner 门槛全部通过，后续会话才可以提出完整 756-case 重训。访问任何已经揭示的
   报告集都需要用户再次明确决定，而且结果绝不能回流用于选择分辨率。

## 完成定义

只有在具备冻结预注册、已核验 subject 隔离、匹配且可运行的代码、smoke 与峰值显存证据、
两个固定 checkpoint 的完整重载、机器可读的门槛决策、测试、产物哈希以及更新后的交接
文档时，本实验才算完成。仅仅“训练跑完”不算完成。
