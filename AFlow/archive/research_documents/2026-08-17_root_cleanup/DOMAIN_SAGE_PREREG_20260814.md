# Domain-SAGE 3D 配对实验预注册（2026-08-14）

状态：代码单测通过、registry/GPU smoke 尚未运行；本文在正式训练前冻结。  
`authorizes_development_training=true`  
`authorizes_test_access=false`

## 1. 假设与定位

在已经验证的 frozen robust refiner 上，使用仅由训练集拟合的潜在域描述符对 SAGE-3D
adapter 进行多尺度低秩条件调制，可能比同 seed、同训练预算的 plain SAGE-3D 获得更高
的 subject-clean development Dice。

该方法受 DPD-SAM 的 latent-domain conditioning 原则启发，但不是 DPD-SAM 复现：

- 不修改或声称实现论文中定义不完整的 SAM-Med3D CLS token；
- 不使用 DPD 的连续 prompt prefix；
- 不使用条件扩散；
- 域信息只调制现有的安全 residual adapter，且初始化严格为恒等映射。

## 2. 固定数据与防火墙

- train：972 scans / 444 subjects；
- validation：108 scans / 50 subjects；
- T1ce 单模态，WT=`{1,2,3}`，raw label 4 排除；
- GT-centered N4 128³ refiner cache；
- coarse input 是冻结 SAM-Med3D 的三次 GT-derived corrective-click logits；
- train/validation subject overlap 为 0；
- cache 必须声明 `test_access=false`；
- 270-case test、test manifest 和 test evaluator 不允许作为本实验入口。

因此本轮仍是 **GT-centered oracle development experiment**，不是自动定位、official15
baseline 或 test 结果。

## 3. Fit-only latent-domain registry

1. 使用冻结 robust seed-20260815 base；checkpoint SHA-256 固定为
   `c5d2b48db2666f097cb0d86e78c20ea95d163c07f98401714d25d755e99aceb0`。
2. 描述符输入只有 T1ce；第二个 coarse 通道恒为 0，不输入 label、GT mask 或 coarse logit。
3. 取冻结 base SegResNet 最深层特征做 global average pooling，预期维度 128。
4. 只用972个训练描述符计算逐通道 mean/std，并用 K-means++ 拟合 `K=4`、seed
   `20260818` 的 centroids。
5. validation 描述符只在 centroids 冻结后做最近中心分配，不参与归一化或聚类拟合。
6. registry 固化 case order、descriptor、pseudo-domain、centroid、cache/checkpoint/code hash。

## 4. 配对架构

### GPU0：plain control

- 现有 `FrozenBaseSafeAdapter(mode=plain)`；
- adapter 输入五通道：T1ce、coarse logit、frozen-base logit、base uncertainty、
  base/coarse disagreement；
- 输出 `frozen_base_logit + residual`；
- 只训练 adapter，base 全冻结。

### GPU1：Domain-SAGE 3D

- 与 plain 完全相同的 base、五通道输入、SegResNet adapter 宽度和两输出头容量；
- 两层 MLP：128 → 64 → 4，Softmax 输出域 affinity；
- 在 adapter bottleneck 和三个 decoder level 注入四个 low-rank FiLM；
- FiLM rank=4，domain coefficients 全零初始化，故初始调制严格为 identity；
- 第二 adapter head 与 plain 一样不参与最终输出；
- 总损失为 `DiceCE + 0.1 * pseudo-domain CE`。

epoch 0 两臂必须逐指标复现 frozen base validation scan Dice
`0.6990433552267926`；否则 fail closed，不得正式训练。

## 5. 固定训练协议

- GPU mapping：GPU0=plain，GPU1=Domain-SAGE；
- seed：两臂均为 `20260818`；
- full data：972 train / 108 validation；
- batch size 2、workers 4、BF16；
- AdamW，LR `1e-4`，weight decay `1e-5`；
- joint spatial shift max 12；
- coarse independent shift max 4，probability 1.0；
- maximum 20 epochs，patience 8；
- 每 epoch 在完整108例验证；
- 每臂按 scan-macro validation Dice 选择自身 best checkpoint；
- 不调 threshold、TTA、形态学后处理或 ensemble。

## 6. 判定规则

primary comparison：同 seed best checkpoint 的
`Domain-SAGE scan Dice - plain scan Dice`。

本轮只有同时满足以下条件才称为“通过一号 seed 架构门”：

1. scan Dice 配对差至少 `+0.001000`；
2. subject-macro Dice 不低于 plain；
3. 无训练崩溃、空预测异常、域分配 collapse 或 test access；
4. 两个 checkpoint 独立 reload 后指标与训练记录一致。

通过后才补第二独立 seed 和 capacity-matched 非域 FiLM control。未通过则记录为负消融，
不按最高点选择、不继续搜索 K、rank、loss weight 或 epoch。

subject-level 差异需以50个 subject 为单位 bootstrap 100,000 次，报告95% CI；CI跨0时
不得宣称统计显著。

## 7. 预训练代码冻结

| file | SHA-256 |
|---|---|
| `sammed3d/t1ce_wt_domain_sage.py` | `6bd3c00703259921517d149702986b4c7dc8f17b54b4515944d70fd2766ca94e` |
| `scripts/build_t1ce_wt_domain_sage_registry.py` | `e58a4850c5abf17acaf4bac93dc3ed8f2bfab958bb20d326e6abd9dfef539bc3` |
| `scripts/train_t1ce_wt_domain_sage.py` | `07a67718d8f9bc03438a3f53ec680d2b564c2be6f596509a06cba4cb7fdebe50` |
| `scripts/run_t1ce_wt_domain_sage_pair_20260814.sh` | `50b9a1ec551101b1118843ff5fee2e97af4a8379f813184f1e5755cf8cb16e9e` |
| `tests/test_t1ce_wt_domain_sage.py` | `a2bd3618557b476c4a71999f5fd815fc390012bb03389069c0f5d2be2395ed7f` |

Targeted CPU acceptance before registry extraction: `9 passed`（Domain-SAGE + existing
safe-adapter tests）。正式训练前仍需真实双 A100 smoke 和全仓回归。
