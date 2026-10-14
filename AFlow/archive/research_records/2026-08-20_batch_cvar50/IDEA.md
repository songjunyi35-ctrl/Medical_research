# Experiment idea: Batch-CVaR50病例尾部聚合

Status: completed fixed-validation negative; stopped without test access

## 1. 现在的问题是什么？

KD200 fixed validation Scan/Subject/Point=`0.734384308/0.724434143/0.935185`，相对matched control平均Scan只提高`0.00077865`。最差10例占全部负收益约63%，去掉这10例后平均增益会从`0.000779`升到约`0.001464`。

VM Dice在boundary-top27产生`+0.000903`局部增益，但KD200-bottom27为`-0.001495`，说明只修改困难体素不足以修复整例灾难。下一步应改变病例级聚合，而不是继续添加体素loss。

## 2. 一句话直觉

每批8个病例像8名学生考试：不按8个人平均分教学，而优先复盘本批loss最高的4人；同时不只盯最差1人，避免单个异常标注控制训练。

## 3. 为什么选择这个idea？

- 2025《Top-k Bottom All but σ Loss Strategy for Medical Image Segmentation》明确把image-level Average Top-k用于医学分割，动机是病例平均会稀释hard cases；论文也警告过小k会导致不稳定。
- 2026 DuetFair/FairDRO同样指出患者级high-loss failure会被均值掩盖，但完整FairDRO依赖临床subgroup+dMoE，且论文没有发布代码、KL半径或求解细节，当前不能忠实实现。
- 当前错误审计直接支持case-tail：少数case贡献大部分负质量，而形态规则不能解释ensemble互补。
- LoMix/DCD/KAS/hACE/VM Dice都是体素、feature或语义层约束；它们的负结论没有测试病例级risk aggregation。
- `top4/8=50%`是batch8的保守工程适配。来源论文的10%在batch8会退化为top1，风险过高；本项目不在validation上比较多个比例。

Primary source: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12428111/>

## 4. 具体怎么做？

只训练一个CVaR50 arm，复用已经完成且证据完整的KAS teacher-free control。

输入T1ce经过原SegResNet得到raw foreground logit。对batch内每个volume独立计算现有loss：

- per-case Dice：与MONAI `DiceLoss(sigmoid=true,squared_pred=false,batch=false)`相同；
- per-case CE：`BCEWithLogits(reduction=none)`后每个volume对全部体素平均，再乘`0.1`；
- `l_i=Dice_i+0.1*CE_i`。

Uniform control为`mean(l_1...l_8)`；CVaR50 arm固定取`torch.topk(per_case_loss,k=4,largest=true,sorted=true).values.mean()`。top-k离散索引不反传，选中的四个loss保留正常梯度。同一case若被replacement sampler在batch内重复抽中，按两个draw分别参与排名，不去重。

没有teacher forward、额外参数或推理变化。

## 5. 唯一改变什么，哪些不变？

唯一变化：病例loss从8例均值改为最高4例均值。

固定：

- KD200起点 SHA `56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7`；历史validation已选且后来访问test，本实验不使用test调参。
- matched control：`workspace/BraTS/sammed3d_t1ce_wt/plain_iad_kas_20260820/control_seed20260860`，TRAIN_DONE SHA `ce987642413fbf86ccbb45ec263851b8a12cfbec6f21d2325521529b583c6392`。
- seed20260860、batch8、weighted replacement、756 draws、drop_last、94 steps、shift8。
- fresh AdamW、LR`1e-5`、weight decay`1e-5`、betas`0.9/0.999`、eps`1e-8`、clip5。
- global201–232、checkpoints208/216/224/232。
- CVaR arm必须与archived control的initial state、first batch raw/augmented tensors、201–232 case-order和CUDA RNG hashes逐项相同。
- validation108、threshold0、无SAM/TTA/postprocessing；部署4,700,897参数、1×forward。

## 6. 成功、失败和停止规则

Selection：只按CVaR arm在208/216/224/232中的Scan Dice最大，Subject exact tie，最终完全同分选更早epoch。tail或Point不得选择checkpoint。

Promotion全部必须通过：

1. selected CVaR Scan/Subject严格超过same-epoch archived control；
2. selected CVaR Scan/Subject严格超过KD200；
3. Point至少101/108；
4. 相对control与KD200的paired Scan/Subject 95% CI下界均`>=-0.003`；
5. KD200-bottom27 CVaR−control mean Dice严格`>0`；
6. boundary-top27 CVaR−control mean Dice`>=0`；
7. implementation、uniform parity、pairing、reload、deploy、memory和access门全部通过。

Technical stop：per-case mean不能在`1e-7`内匹配当前DiceCE loss/gradient；top4不是四个最大loss；selected loss无梯度；任一loss/gradient NaN/Inf；配对hash不同；formal peak allocated超过24GiB；外推超过45min；checkpoint/deploy不等价。

任一promotion失败即停止，不搜索top比例、curriculum、loss、LR、epoch、threshold或postprocessing，不访问test270。

## 7. 代价

- 只训练一个新arm；每step仍1×student forward，仅增加长度8的top-k。
- 预计18–22min、约20.1–20.8GiB，24GiB硬门。
- 四个plain checkpoint与报告约75MB；无新cache。
- 推理仍4.70M参数、1×forward。

## 8. 最终结果（完成后填写）

- Validation：candidate按Scan选择epoch208，Scan/Subject/Point=`0.731800870/0.721921887/0.925926`。同轮新uniform为`0.733194361/0.723333983/0.944444`，KD200为`0.734384308/0.724434143/0.935185`。
- Test：未授权、未访问。
- Tail mechanism：KD200-bottom27 candidate−uniform=`-0.002274960`；boundary-top27=`-0.001550842`，两门均失败。
- Cost：两臂formal约`1107.86/1106.53s`，双卡墙钟18.46min；每臂peak allocated约20.29GiB；workspace约145M；推理1×。
- 是否支持直觉：不支持。candidate相对uniform Scan/Subject=`-0.001393491/-0.001412096`且95% CI整体低于0；tail也恶化。
- 下一步：停止当前top4/8适配，不搜索比例、curriculum、smoothing、loss、LR或epoch。近期所有loss/teacher约束都未超过KD200，下一步应离开loss轴，重新核验局部原生分辨率或架构信息瓶颈。
