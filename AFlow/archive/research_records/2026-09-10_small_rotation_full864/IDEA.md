# Experiment idea: 仅在旋转开发通过后，全864固定配对重训

Status: conditional planned, not admitted.

## 1. 现在的问题是什么？

旧同internal reused test270/119 overall Scan第一0.7572696616942132（Full972KD168八镜像）；1次前向Scan第一0.7520031755617648。当前开发fit684与test数据规模/队列不同，不能直接超榜。旋转四模型尚无开发结果。

## 2. 一句话直觉

只有小考两个种子都可靠，再用全部已授权教材训练一对模型，参加同一场固定考试。

## 3. 为什么选择这个idea？

条件是旋转主研究完整联合门通过，不靠训练损失猜晋级。全864增加覆盖但不保证涨分；一手增强依据沿用 https://arxiv.org/abs/2010.13372 ，不称全数据重训收益已有证明。旧弱弹性全864分支从未启动，其主实验负结果不能冒称此旋转分支失败或成功。

## 4. 具体怎么做？

授权fit684+dev180合并成864 scans/391患者，从头seed2026091071训练control/rotation两臂168轮。若主研究选native_rotation，两臂均原native抽样；若选subject_rotation，两臂均患者等概率抽样。候选仅追加已固定20%/三轴±15°输入网格旋转，控制不加。原4.70M网络、主Dice加0.1交叉熵，无教师。RC辅助标签以零占位且权重0，须先验证所有梯度等价。原dev180从这一步开始是训练，不再作为选择/验证。
完成后单次固定test270四输出：control/rotation各identity和固定八镜像，镜像逆变换raw logit按既定顺序float32累加/8。掩膜阈值0、点为各自raw-logit最高分位置，不换定位定义。8次前向中的identity复用首视图，共4320前向。

## 5. 唯一变化与固定项

两臂仅旋转有无；采样策略由先前开发冻结规则唯一选择后两臂相同，初值/seed/数据/168轮/基础随机流/优化器/输出全部匹配。旧confirmation108/53继续排除，fixed validation/outer/external不读。fit合并manifest SHA预期d92619562fb301509f9668873684a3a7f57b0edee1df1fc35e65cea43bdbba52，checkpoint/图像标签/cache-index/test清单/历史榜首报告在test前各自绑定SHA。

## 6. 成功、失败和停止规则

必须主研究selected_arm合法且独立VERIFICATION通过，完整168重放通过；新fit8双臂GPU smoke初态/原RNG/导出exact且<24GiB。时间准入=最慢主研究168轮×1.1+1200秒，须能在21:39:55 UTC前完成；计算监督器21:24:55前停，保留归档。
test固定患者簇bootstrap10000次、seed2026091032，原270每扫描float32 Dice口径及患者均值。报告相对旧overall和matched control的配对区间。点估计领先、95%区间支持与联合科学门分开：候选八镜像Scan与Subject须超过旧overall及同八镜像control、Point至少265且不低于control，两项旧榜对比区间下界>0才称明确统计支持；1次前向Scan>0.7520031755617648另列，不据其选择视图。具体计算门继承已测试通用helpers，test前冻结。如继承门含更详细规则，须在访问前明确核对，不以结果后解释替代。
只有一份新test面板，所有四行正/负均归档并更新两榜。不按test调角度、seed、epoch、阈值、融合或后处理。不能称blind/external/官方挑战SOTA；历史test结果已知、流水线test-informed背景保留。未过主开发门/预算/工程则不执行并记录未准入。

## 7. 成本与风险

另双卡约2–2.3h、4–4.6GPUh，两个模型与恢复点<200MiB，无新像素cache。保存四组270例128³二值mask共270MiB，推理1×/8×明确区分。全部授权数据训练会消耗原dev角色，不能再称模型没见过dev；如本分支启动，备用fit研究本窗口不再执行。训练镜像官方字节parity未证明限制保留。

## 8. 最终结果（完成后填写）

未准入/未执行；没有新full模型或test结果。
