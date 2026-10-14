# 固定Flip8跨cohort只读综合：分析冻结

状态：`post-access descriptive synthesis`；三个输入报告均已揭示。本记录不声称预注册新实验，不运行模型、不选择模型、不调参。

## 1. 现在的问题是什么？

Full972 KD168的固定Flip8在reused internal test270把Scan Dice从`0.751760`提高到`0.757270`（`+0.005510`，配对95% CI全正），在Africa95从`0.523003`到`0.526438`（`+0.003435`，CI跨0），在刚完成的CPTAC33从`0.841331`到`0.844577`（`+0.003246`，CI全正）。这些数字散在三份报告里，容易把“每处点估计都正”误写成“所有域都确定有效”。

## 2. 一句话直觉是什么？

Flip8测试时增强（Test-Time Augmentation，TTA）像让同一位医生把MRI从八个镜像方向各看一次，再把八份答案翻回原位取平均；如果模型对摆放方向有偶然偏差，平均可能抵消它。

## 3. 为什么现在做它？

CPTAC33刚提供第三个已运行Flip8的独立队列视角，可以回答“方向是否一致”和“哪处不确定”，但不能再用这些已见结果选择视图或权重。旧的等变一致性训练双复制失败，只否定“用那一版训练损失把8×压回1×”的实现；它不否定部署时直接平均八个固定视图。

## 4. 具体怎么做？

只读三份已冻结raw JSON：对同一Full972 KD168 checkpoint，取原图一次输出（identity）和三轴反射全部八个视图逆翻后float32等权平均（Flip8），列出每个cohort的绝对Scan/Subject Dice、Point hit、Flip8−identity差值与原报告配对95%区间。输出机器JSON和图，不重新加载图像或checkpoint。

## 5. 只改变什么，哪些不变？

- 唯一分析变化：把三个既有cohort并排展示。
- 固定：checkpoint SHA、128³输入、阈值0、八视图集合/顺序/等权raw-logit融合、Point定义、无Segment Anything Model（SAM）/后处理/模型ensemble、原bootstrap结果。
- 不允许：合并cohort排名、重新bootstrap换seed、挑病例、调视图/权重/阈值、再次评测。

## 6. 怎样算成功/失败？

这不是promotion实验，没有晋级门。描述规则在生成前冻结：三处差值都大于0才写“点估计方向一致”；只有某cohort配对95% CI下界大于0，才写该cohort的Scan改善区间支持；任一差值小于等于0即写“方向未跨三处一致”。无论结果如何，都不能写普遍有效或因果结论。

## 7. 代价是什么？

新增训练0、GPU 0、模型forward 0；JSON/图预计低于1 MiB。若部署Flip8，每例locator计算和延迟约为identity的8倍，参数/checkpoint和峰值显存近似不变。三次历史评测的计算已经发生，不能把只读综合误写成零部署成本。

## 冻结输入

- internal270 `RAW_TEST270.json` SHA-256 `29efc557aa12b6f82fd079edd84cc3c8c1e5ad4536880c016b664932234f45ad`
- Africa95 `RAW_EXTERNAL95.json` SHA-256 `c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3`
- CPTAC33 `RAW_CPTAC33.json` SHA-256 `4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd`

UCSD75没有运行固定Flip8，明确不填补、不推断。
