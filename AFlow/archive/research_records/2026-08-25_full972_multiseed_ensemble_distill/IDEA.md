# Experiment idea: Full972多seed等权logit ensemble并蒸馏回单模型

Status: planned and frozen before new-seed implementation, model construction, or validation/test access.

## 1. 现在的问题是什么？

Full972 random-init KD seed20260932在fixed validation108达到`0.740013/0.731266/103/108`，在用户授权failed-gate reused test270达到`0.751760/0.737327/262/270`。但只有一个随机seed，无法区分稳定机制和幸运轨迹。

## 2. 一句话直觉

同一道题让三位独立学生作答，再平均他们的信心，通常比只信一个人更稳；如果三人组有效，再把共识教回一个学生。

## 3. 为什么选择这个idea？

- Full972相对matched Full756 validation Scan提高`+0.010943`且95% CI整体为正，数据机制成立。
- 当前剩余风险是随机初始化方差，而不是病例数、预算或late continuation。
- Cross-seed parameter soup曾灾难性失败，但它平均的是不同盆地的参数；本方法平均输出logits，不要求参数位于同一盆地，因此旧负结果没有否定它。
- 旧history/EMA logit ensemble已经证明功能互补，并成功蒸馏为KD200；同样路线可用于更强Full972 seed。

## 4. 具体怎么做？

阶段A：完全复用Full972 final协议，新跑seed20260940和20260941。两者均从随机初始化训练到inner已冻结的epoch168，每轮972 requested / 968 consumed / 121 steps，使用相同history/EMA voxel-KD teacher。

阶段B：冻结现有seed20260932与两个新checkpoint，raw logits按`1/3`等权平均；threshold0，point取平均logit argmax，无TTA/SAM/postprocess。只报告fixed validation108，不用validation搜索权重或seed。

阶段C：只有等权ensemble通过validation门，才把三模型平均logit作为teacher，训练一个新Full972 student到冻结epoch168，部署恢复1×。

## 5. 唯一变化与固定项

- Seed复现唯一变化：随机seed20260940/20260941。
- Ensemble唯一变化：三份冻结raw logits等权平均。
- 固定：Full972病例、cache、原生体积权重、4.70M模型、epoch168、968 draws/121步、旧teacher、loss、LR、增强、threshold0、无TTA/SAM/postprocess。

## 6. 成功、失败和停止规则

- 两个seed必须完整168轮、初始state彼此不同、所有工程门通过；不能按validation/test挑seed。
- Fixed ensemble公式必须是seed32/40/41各`1/3`。
- Ensemble validation必须严格超过seed32 Scan/Subject，Point>=101；相对seed32的Scan/Subject paired 95% CI下界>=-0.003；才授权阶段C。
- 阶段C student必须在validation同时超过seed32和KD200 Scan/Subject、Point>=101及四CI安全门，才允许任何post-test探索性评估。
- 任一训练>6h、peak>24GiB、NaN/Inf、teacher gradient、checkpoint reload/hash失败立即停止。

## 7. 成本与风险

- 两个新seed两卡并行约3.6h，每卡20.3GiB；新增checkpoint/report<500MB。
- 三模型ensemble推理3×，只用于teacher/研究。
- 若蒸馏，再约3.6h，最终1×；不建立完整logit cache。
- 最大风险：seed32可能是幸运轨迹，新seed较弱导致等权ensemble反而下降。

## 8. 最终结果（完成后填写）

- Seed replication：待运行。
- Ensemble validation：待运行。
- Distilled student：仅ensemble过门后允许。
