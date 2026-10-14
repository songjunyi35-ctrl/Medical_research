# Experiment idea: KD200–DMRFB216 same-basin weight interpolation

Status: validation Dice positive but Point promotion failed; stopped without test access

## 1. 现在的问题是什么？

DMRFB216 fixed validation超过same-parameter control：Scan/Subject `+0.000245/+0.000146`，并改善boundary-top27 `+0.000107`和KD200-bottom27 `+0.001184`。但它相对KD200 Scan `-0.000142`且Point少1例，因而不能进test。

需要判断能否在一个模型内保留KD200的Scan/Point稳定性，同时获得DMRFB的Subject/tail收益。

## 2. 一句话直觉

两端是同一锅底训练出来的两份配方；按比例混合参数，尝试在KD稳定性和DMRFB感受野收益之间找到折中。

## 3. 为什么选择这个idea？

- KD200可确定性装入相同`dmrfb_dilation2`架构，adapter保持初始zero projection，epoch0 logits与plain KD200精确相同。
- DMRFB216从这个精确函数起点继续训练；不是过去失败的cross-seed soup。
- 两端90个state key完全一致；base network relative-L2仅`0.01023`，整体cosine约`0.99155`；alpha0.5 BF16 forward finite。
- 该方法不再训练、不改架构、不增加forward，成本低，且只使用development validation选alpha。
- 它是项目same-trajectory插值，不声称新论文复现。

## 4. 具体怎么做？

- Endpoint0：从KD200 checkpoint构建`dmrfb_dilation2`模型；adapter使用construction seed20260901和zero projection。
- Endpoint1：DMRFB candidate global epoch216 checkpoint，SHA `df53d9bd61f1489746ae1240fffba1f8a3897d0de1416122a389e1d420def684`。
- 对每个浮点state tensor：`theta_alpha=(1-alpha)*theta0+alpha*theta1`。
- 非浮点tensor必须两端逐元素相同才可复制；否则技术失败。
- 固定interior grid `alpha={0.25,0.50,0.75}`；不评其他alpha。
- 每个checkpoint strict reload，并验证同一固定输入logit finite。
- 推理仍为DMRFB dilation2单模型raw logit阈值0。

## 5. 唯一变化与固定项

唯一变化是alpha。架构、state key、dilation、参数量4,757,985、validation、threshold、SAM/TTA/postprocessing全部固定；没有训练、teacher或新cache。

Endpoint metrics只作冻结比较：KD200=`0.734384308/0.724434143/0.935185`；DMRFB216=`0.734242477/0.724742362/0.925926`。

## 6. 成功、失败和停止规则

Selection只在三个interior alpha中按Scan最大、Subject exact tie、最终同分选较小alpha。

Promotion全部必须通过：

1. selected Scan和Subject严格超过KD200；
2. selected Scan和Subject严格超过DMRFB216；
3. Point至少101/108；
4. 相对KD200和DMRFB216的paired Scan/Subject 95% CI下界均`>=-0.003`；
5. checkpoint generation/reload/state geometry/hash/access门全部通过。

Boundary/bottom27只报告，不选alpha、不作为promotion，避免继续在同一validation上堆机制门。

若失败，不增加alpha、改成logit ensemble、调threshold或postprocessing，不访问test。

## 7. 代价

- 无训练；生成3个约19MB checkpoint。
- Fixed validation约3×单模型forward。
- 部署一个4,757,985参数模型、1×forward；与DMRFB相同。
- 无新cache，磁盘增量约60MB。

## 8. 最终结果（完成后填写）

- Validation：selected alpha0.5 Scan/Subject/Point=`0.734683010/0.724934341/0.925926`，同时超过KD200和DMRFB216的Scan/Subject，但Point仅100/108。
- Test：Point门失败，未授权、未访问。
- Cost：无训练；生成3个checkpoint约55M，CPU strict-forward约数分钟；validation约19秒GPU；部署4,757,985参数、1×。
- 是否支持直觉：Dice折中直觉支持。相对KD Scan/Subject=`+0.000299/+0.000500`，相对DMRFB=`+0.000441/+0.000192`；boundary/bottom27相对KD也为正。Point稳定性没有恢复。
- 下一步：停止当前插值，不换alpha或借KD点位。若用户未来明确将目标改为“Dice优先、Point可少1例”，可把alpha0.5作为development-selected单模型候选；在当前冻结目标下不得test。
