# AFlow / BraTS Claude 接续交接

> **Superseded handoff notice（2026-08-10）：**本文件已由
> `AFLOW_BRATS_HANDOFF_20260807.md`、
> `AFLOW_BRATS_HANDOFF_20260810_ADDENDUM.md` 和 `T1CE_MAINLINE.md` 覆盖。
> 当前唯一活跃科研主线是 T1ce -> WT={1,2,3}；以下内容仅作历史记录。

状态时间：`2026-08-05 UTC`

## 0. 必须先遵守的指令

1. 先读 `AFLOW_BRATS_HANDOFF.md` 的第 0 节，再读第 40/41 节和第
   61--67 节。冲突时以第 0 节最上方、最新指令为准。
2. **永久禁止再访问 270-case test**：不读 manifest、NIfTI、逐例报告，
   不运行 test evaluator，不用历史 test 结果做任何模型选择。
3. 当前只能使用已冻结的 `972 train / 108 validation / 50 validation subjects`
   clean-development 协议。
4. 禁止 threshold/TTA/morphology/connected-components/postprocess 搜索，
   禁止 full image-encoder unfreeze，禁止为这个 arm 临时改 LR、解冻层数、
   initialization、augmentation、epoch cap 或 patience。
5. 不要 `git reset`/`git clean` 或批量删除。仓库存在大量已有 modified/deleted/
   untracked 文件，它们不是本次交接授权清理的内容。

## 1. 一句话现状

Matched rank-0 baseline **已经完整训练 50 epochs**，LoRA-vs-rank-0 的
clean-validation 成对分析也已完成。现在唯一待跑的科研任务是预注册的
`image_encoder.blocks.8--11` partial-unfreeze + discriminative-LR arm。它的代码、
CPU preflight 和工程门已通过，但 GPU smoke 和 formal training 都尚未开始。

## 2. 我实际做了什么

### 2.1 重新读取与核验科研边界

- 严格先读了 `AFLOW_BRATS_HANDOFF.md` 第 0 节，再读了第 40/41 和
  61--67 节。
- 本次盘点只读取 train/clean-validation 产物、代码、preregistration 和
  engineering audit；没有读取或枚举 270-case test 数据。

### 2.2 确认 rank-0 不是待跑任务

直接检查了：

`workspace/BraTS/sammed3d_t1ce_wt/fv_rank0_matched_aug_train972_val108_e50_s20260725/history.csv`

核验结果：

- CSV 为 `51` 行：1 行表头 + 50 个完整 epoch。
- history SHA-256：
  `3e1c87e93ecf2881b23657c2562d15fd541ab4b6588a4b151315332551700c80`。
- selected epoch=`32`，trainer validation C3=`0.49248712568509356`。
- epoch 50 train/validation C3=`0.5927532873230933 / 0.4777732711388833`。
- selected checkpoint SHA-256：
  `7a25d0268799bdfffc626eacecb2a319a51d8127460143e7c6af3e8d4b532cf9`。
- independent reload clean-validation：scan-macro C3=`0.4816013968`，
  subject-balanced C3=`0.4785050270`。

### 2.3 确认 LoRA-vs-rank-0 开发证据

- LoRA selected checkpoint 是 epoch 33，SHA-256：
  `c1d1bd79791aa7beeffc541095472dec968630cc70ea7967128c86552bd7f1ec`。
- LoRA independent reload clean-validation：scan-macro C3=`0.5244305128`，
  subject-balanced C3=`0.5214101623`。
- 与 rank-0 相比，subject-balanced C3 delta=`+0.0429051353`；50-subject paired
  bootstrap 95% CI=`[+0.0258629117,+0.0630796008]`，42/50 subjects 改善。
- 这只是同一 clean validation 上、单 seed、validation-selected 的 conditional
  development contrast，不是 confirmatory/external 证据，也不是确定性的纯 LoRA
  因果效应。

主证据：

- `workspace/BraTS/research_audits/fv_lora_vs_rank0_matched_val108_conclusion_20260724.md`
  当前 SHA-256：
  `3482801189e8055140aacadc70cd551ab673351e245b38c275eaa0235835fdbf`。
- `workspace/BraTS/research_audits/FV_LORA_VS_RANK0_MATCHED_VAL108_CPU_PAIRED_20260724.json`
  当前 SHA-256：
  `e9918aedb8825ca87147d61f321c03f296e0014ddf07426f7017e08770ce01ee`。

### 2.4 确认 partial-unfreeze 工程门仍然完整

2026-08-05 重新计算后，preregistration、helper、trainer、launcher 和 tests 的
SHA-256 与 engineering gate 记录全部一致，没有发现漂移。

Engineering gate 的原记录是：修复 model/optimizer 构建后 RNG reseed 之后，
focused tests `24/24 passed`，更宽的 CPU/synthetic contract suite `41/41 passed`，
CPU full preflight 通过且 `output_created=false`。本次只重新核验了文件哈希，
没有冒充重跑这些测试。

核心哈希：

| Artifact | SHA-256 |
|---|---|
| human prereg MD | `111968f92c09f35272a5ddd35741c3ad1a5b2803a09f307b2a6993473e156156` |
| machine prereg JSON | `b55e763c4ab422a8b38d49df7fe6dde9e95af8d5e154d485eaef5fdcc97a5f91` |
| helper | `dce2accc8d950ad171d4dc2417dead153ff2b74a78f007725dcb6ca5e5e32f07` |
| CPU preflight | `9b0af7a5cbddd99264d0037bee5e796c5b41254ba1ed1622053afa1dea6a1efa` |
| trainer | `643c1e9124986ac4d057be3779d7eea5ff7b0e99009cea1361feb496f90cc20e` |
| formal launcher | `7034d7e978fdcf248d53de656482b9d96e78c9279bcf5c9a8d0f52847e507d33` |
| smoke launcher | `5205a6b7e368777fe158fa2aa89453dc83af059599cc81f15263e16c8637ac24` |
| engineering gate | `afc25e55b81fa3d406a6037f261e54dd44a6596ddb8d65feda065d17c466d39a` |

### 2.5 H100 / worker 尝试

- 2026-07-25 engineering gate 记录过 A10/Arnold control-plane 分配失败，没有产生
  smoke/formal output。
- 后续会话中按用户要求尝试使用 H100 继续：workspace worker 曾进入
  Ready，但 MLX login/SSH 未得到可用的训练 shell；public Arnold 不可用必要的
  liveaim volume，private Arnold 还需指定 cluster。这次尝试没有运行 trainer，
  没有新建任何训练产物。
- 2026-08-05 使用非沙箱的只读 `mlx worker list` 再次核验：当前列表为空，
  没有 active worker。当前 Codex 执行环境也没有暴露 `/dev/nvidia0`。
- 未发现正在运行的 BraTS trainer/evaluator。

## 3. 待跑 arm 的唯一冻结配置

Run identity：

`fv_partial_last4_disclr10_train972_val108_e50_s20260725`

- 从 untouched `sam_med3d_turbo.pth` 新鲜加载，不从 LoRA/rank-0 selected
  checkpoint 初始化。
- untouched checkpoint SHA-256：
  `899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`。
- LoRA 关闭，target token 关闭。
- 只解冻 image encoder blocks `8,9,10,11`：`28,367,616` params，
  AdamW LR=`1e-5`，WD=`0.1`。
- prompt encoder + mask decoder：`7,584,304` params，AdamW LR=`1e-4`，
  WD=`0.1`。
- trainable/model=`35,951,920/100,509,232`。
- batch 1，accumulation 2，BF16，2 workers，train seed `20260725`，
  validation/click seed `20260723`。
- 最多 50 个完整 epochs；epoch 20 前不允许 performance early stop；之后只在
  20 个连续完整 epochs 没有 validation C3 strict new high 时停止。
- selected checkpoint=所有 completed epochs 中 trainer validation final-click scan-macro
  C3 的 strict argmax，并列保留更早 epoch。

唯一准据：

- `workspace/BraTS/research_audits/FV_LAST4_PARTIAL_UNFREEZE_DISCLR_PREREG_20260724.md`
- `workspace/BraTS/training_protocols/fv_last4_partial_unfreeze_disclr_prereg_20260724.json`
- `workspace/BraTS/research_audits/FV_LAST4_PARTIAL_UNFREEZE_ENGINEERING_GATE_20260725.md`

## 4. 当前输出状态

以下两个路径在 2026-08-05 均**不存在**：

```text
workspace/BraTS/sammed3d_t1ce_wt/smoke_fv_partial_last4_disclr10_2train_2val_e1_s20260725
workspace/BraTS/sammed3d_t1ce_wt/fv_partial_last4_disclr10_train972_val108_e50_s20260725
```

因此：

- GPU smoke 没有跑过；
- formal preflight 没有在 GPU 上跑过；
- formal training 没有启动；
- 没有 partial-unfreeze 科研指标可以报告。

## 5. Claude 的严格接续顺序

### Step 1：获得一张真正可用的 H100/BF16 worker

1. 先运行 `mlx worker list`，排除 orphan allocation。
2. 进入 worker 后确认工程根目录、venv、untouched checkpoint 和 train/validation
   manifests 都可见。不要检查 test 路径。
3. 运行 `nvidia-smi` 和：

```bash
cd /mlx_devbox/users/songjunyi.666/playground/AFlow
bash scripts/run_worker_research.sh probe
```

### Step 2：只跑唯一授权的 GPU smoke

```bash
cd /mlx_devbox/users/songjunyi.666/playground/AFlow
bash scripts/run_sammed3d_t1ce_wt_partial_unfreeze_smoke.sh
```

必须同时确认：

- 只有 2 train / 2 validation / 1 epoch；
- loss、两个 optimizer groups 的 gradients 和 metric 都 finite；
- 记录 H100 显存和吞吐；
- checkpoint metadata 完整；
- save + strict reload 成功；
- smoke metric 只是工程信号，不能用它改变任何科研配置。

如果 smoke 失败，先诊断工程/环境问题；不得绕过 smoke 启动 formal。

### Step 3：smoke 通过后，在同一前台 worker 跑 formal

```bash
cd /mlx_devbox/users/songjunyi.666/playground/AFlow
bash scripts/run_sammed3d_t1ce_wt_partial_unfreeze_formal.sh preflight
bash scripts/run_sammed3d_t1ce_wt_partial_unfreeze_formal.sh run
```

不要放到未受监控的后台 watcher。`run` 会再做一次 fail-closed preflight，然后
`exec` 成科学训练进程。formal output 必须在首次启动前不存在/为空。

训练中只监控：进程存活、finite metrics、显存/吞吐、atomic epoch 完成和冻结
early-stop 规则。不根据中间 validation 曲线改配置。

### Step 4：formal 完成后只做一次 108-case clean-validation independent reload

1. 严格 reload trainer 选中的 checkpoint，不评估每个 epoch，不重选 epoch。
2. 使用同一 108 cases / 50 subjects、validation seed `20260723` 和已有 paired
   evaluator contract。
3. 主比较是 `partial-unfreeze - selected LoRA` 的 subject-balanced C3，LoRA
   reference=`0.5214101623252374`。
4. 做 10,000 次 subject-level paired percentile bootstrap，seed `20260724`。
5. 只有 point delta `>0` 且 95% CI lower bound `>0` 才算 development GO。

如果任一条不成立：NO-GO，不扩 seed、不调参、不访问 test。即使通过，也只是
允许讨论新的 development confirmation preregistration，**仍不授权 test**。

## 6. 仓库与安全注意事项

- 当前 git HEAD：`3f45721` (`main`)。
- BraTS 研究文件多数是 untracked，同时还有很多与此任务无关的已有
  modified/deleted 文件。不要为了“整理”而覆盖或删除。
- 本次没有 commit，没有改动 trainer/preregistration，只新建了本交接文档。
- 若发现 prereg 绑定哈希不一致，必须 fail closed；不要为了开跑而改哈希。
- 训练器没有经审计的 exact-resume contract。若 epoch 1 已完成后中断，不得静默
  从 untouched checkpoint 重跑并挑更好的 attempt；按 prereg 的 amendment 规则处理。

## 7. 与 BraTS 科研无关、但本会话已做的 checkpoint 盘点

这些是用户另外询问的 Qwen 训练产物，不要和 BraTS 权重混淆：

1. 用户最初给的路径
   `20260724_qwen36_27b_fs_cgvr61d_n10_gold_1x_lr5e6_s150` 本身不存在。
   实际高度匹配目录是
   `20260724_qwen36_27b_fs_corhard_cgvr61d_n10_gold_1x_each_lr5e6_s150`。
   其 v0/v1 都有 checkpoint-50/100/150；各自 checkpoint-150 有 12 个
   safetensors shards，约 51G，iteration=`150`。
2. 目录
   `20260727_qwen36_27b_fs_gambling_corhard_cgvr61d_gold_1x_each_lr5e6_s150`
   下有：
   - `v0-20260728-022101`：checkpoint-50/100/150；
   - `v1-20260728-101130`：checkpoint-600 存在，`config.json` 存在，
     `latest_checkpointed_iteration.txt=600`，12 个 safetensors shards，目录共 21 文件。
3. 用户给出的 vLLM 自动选择脚本不是固定选 v1；它按 run 目录
   `st_mtime_ns` 取最新，并在其内取数字最大的 checkpoint。盘点当时会选
   `v1-20260728-101130/checkpoint-600`；如果以后 v0 的目录 mtime 被更新，
   选择结果可能改变。

## 8. 最短接手提示

> 先读 `AFLOW_BRATS_HANDOFF.md` 第 0 节，再读 40/41 和 61--67。不得再访问
> 270-case test。rank-0 50 epochs 已完成，不要重跑。当前只需在一张可用 H100
> 上先跑冻结的 2/2/e1 partial-unfreeze smoke，通过后再以前台方式跑 formal
> preflight 和 formal run。不改任何预注册超参，不绕过 smoke，不做 test。
