# Plain locator 100→200 epoch / late cosine 预注册（2026-08-18）

状态：`FROZEN_BEFORE_TRAINING`

## 1. 科研问题

检验现有 plain locator 的 100 epoch 上限是否截断了学习，以及在不改变模型、loss、数据增强
和 raw-logit threshold 的前提下，100 epoch 后的低学习率 cosine continuation 能否稳定提高
scan-macro、subject-balanced Dice，同时保持 automatic point hit。

本实验不检验 intensity/gray residual；该分支已于 2026-08-17 判负。

## 2. 启动依据

- 旧 development plain 和当前 subject-clean benchmark plain 都在 epoch 99 创新高；相对此前
  最佳分别提高 `+0.003364` 和 `+0.002718`；
- 两条曲线在 epoch 76–100 都产生多次新高，说明存在截断风险；
- 但 2026-08-17 fresh inner plain 在 epoch 79 达峰，并因 20 epoch 无新高于 epoch 99 停止，
  所以“直接以固定 `2e-4` 堆到200轮”没有充分依据；
- 既有训练器没有 scheduler，全程 AdamW `lr=2e-4`，后段验证波动明显。

因此被检验的单一方案是“原100轮 + 100轮 late cosine”，不是任意 epoch/LR 搜索。

## 3. 数据角色与禁止访问

只从 2026-08-17 intensity 实验的 648 fit cases 中重新切分；昨天已揭示的108 inner-tune
完全不进入本轮 source role。

| role | cases | subjects | SHA-256 |
|---|---:|---:|---|
| fit | 540 | 249 | `842be9e0d56ba8848bcb913d54808f9e2b468cc0f904f45f1ed91f1452d113f5` |
| fresh tune | 108 | 52 | `50270be1ecb50a3283e844b2916ae9ee8369989d6f57273706595f3a9cd7d7d2` |

case overlap = 0，subject overlap = 0。split manifest SHA-256：
`f9f1967eab5e6ad1105c7147c2bd0c0eee227b2473891d414c3e9196da6a7836`。

builder manifest 中的 `purpose` 字段沿用了 intensity builder 的历史字符串；本文件在任何训练
前冻结本 split 的真实用途，角色、seed、SHA 和隔离检查不受该描述字段影响。

本轮 inner 决策前禁止访问：

- 旧/current 108-case outer selector；
- 108-case confirmation；
- 270-case test；
- 2026-08-17 已揭示的 intensity inner-tune 108。

## 4. 固定模型与训练

- model：plain `AutomaticPromptLocator` / MONAI SegResNet；4,700,897 params；
- input/target：T1ce → WT `{1,2,3}`，raw label 4 excluded；
- geometry：N4 + image-derived brain crop → `128^3`；
- loss：Dice + `0.1 * CE`；
- optimizer：AdamW，weight decay `1e-5`；
- batch：8；seed：`20260823`；augmentation 与现有 plain 完全相同；
- fixed threshold：raw logit `>0`；无CAPG、SAM、TTA、connected components或灰度阈值；
- 完整跑满200 epoch，不以 tune early-stop。

唯一改变是学习率时间表：

- epoch 1–100：固定 `2e-4`；
- epoch 101–200：cosine，从 `2e-4` 衰减到 `1e-5`；
- 不尝试第二套 minimum LR、warmup、restart 或其他 schedule。

## 5. checkpoint窗口与主比较

同一条确定性训练轨迹保存两个互斥窗口：

1. `pre100_best`：epoch 1–100 中 fresh-tune scan Dice 最高；
2. `post100_best`：epoch 101–200 中 fresh-tune scan Dice 最高。

主因果对比为 `post100_best - pre100_best`。两个 checkpoint 都必须独立重载全部108例，
重载与保存时逐病例 Dice 的最大绝对差必须为0。

## 6. inner晋级门槛

以下条件必须全部满足：

1. scan-macro Dice 差 `>= +0.003`；
2. subject-balanced Dice 差 `>= +0.003`；
3. paired subject bootstrap 95% CI lower bound `>= 0`（10,000次，seed `20260824`）；
4. automatic point hit 相对下降不超过 `0.02`；
5. 两份 checkpoint 精确重载，且所有 access flags 为 false。

任一失败：`outer_selector_authorized=false`，停止该方案，不在本 fresh tune 上搜索 epoch、LR、
minimum LR 或 patience。

## 7. 通过后的唯一 outer 操作

只有 inner 全通过后才允许：

1. 锁定 post100 的 epoch `E` 和本学习率公式；
2. 在原756-case train role 上从头训练 exactly `E` epochs，不读取 validation 做checkpoint选择；
3. 对这个单一 fixed-epoch checkpoint 在 current outer selector 上评估一次；
4. 与当前 epoch99 plain 的已保存 selector报告做固定比较。

outer 成功线：scan 和 subject 各至少 `+0.003`，point hit下降不超过`0.02`。即使成功，也只是
新的 development candidate；confirmation/test 已揭示且禁止再次用于选择或宣称新benchmark。

## 8. 证据边界

本实验可回答“late-cosine continuation 是否优于同轨迹前100轮”，不能把收益拆分成纯epoch
效应或纯scheduler效应。真正的外部突破仍需要新的外部cohort。
