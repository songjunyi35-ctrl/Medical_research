# Experiment idea: 开发过门后全授权864例固定重训

Status: conditional_preparation_only_before_stage_a_readout

## 1. 现在的问题是什么？

旧reused test270/119最高整体Scan0.757269662（Full972KD168八镜像，8×），单次前向Scan0.752003176（GCSAM184）；新fit684/dev180的弱形变四臂尚在训练，没有开发成绩。旧native dev Scan0.746762/0.744177。不能挑一个fit种子直接用开发分数宣称超榜。

## 2. 一句话直觉

先通过两次同条件小考确认训练办法，再用全部获准教材从头训练固定模型，最后参加一次事先定好的旧榜比较。

## 3. 为什么选择这个idea？

条件是弱弹性形变在两seed均过联合门；只有这时才有本项目依据。原始[增强消融](https://arxiv.org/abs/2010.13372)支持空间增强假说但不保证单T1ce收益。本阶段检验从684到864训练病例后收益是否保留，不是新强度搜索。旧患者采样单独失败不被推翻，若该抽样层入选则两臂都固定患者抽样。

## 4. 具体怎么做？

把已授权fit684与dev180合成864次/391患者训练清单，排除全部旧confirmation108/53人；原dev此后为训练，不再次用它评分选优。单T1ce固定128³缓存→原增强（候选追加完全相同弱形变）→同两头训练结构、辅助系数0→相同主损失→固定168轮→精确单通道导出。无教师、无预训练旧checkpoint。辅助零权重使切除腔标签不参与主梯度，两臂用全零辅助占位通道，不建864例RC缓存；必须用合成/fit smoke核验其主梯度与真实RC版本等价。

## 5. 唯一变化与固定项

两臂唯一变化=追加弱形变。固定seed2026091031、同初始权重、batch8/972draw/968有效/121步/168轮、AdamW2e−4/wd1e−5、旧学习率、Dice+0.1交叉熵、128³输入、原增强RNG、logit>0/最高logit单点定位。两臂抽样均使用Stage A唯一入选层：native或患者归一native。不改变概率0.2/场7³/最大2输入体素/梯度0.25/边界0/插值。旧脚本不改，由新进程入口覆盖数据类与协议。辅助占位改变对照实现但两臂完全匹配、主任务等价需验证。

## 6. 成功、失败和停止规则

Stage A原四臂所有科学/工程门与唯一选择规则固定于2026-09-10_weak_elastic/IDEA.md。未过则本阶段关闭不执行。启动前独立复算Stage A摘要/簇CI/配对和选择；全新manifest/代码/候选及control身份/输出哈希冻结。只有双卡空闲、预计168轮+测试与归档能在11:28:19 UTC前完成才准入，不缩短轮数。

full864工程门同Stage A：初值、采样、原增强RNG一致，有限损失/标签安全、峰值allocated≤24GiB、导出error0、完整168轮。每臂6h绝对上限且总session截止优先。该阶段没有开发/outer/confirmation/外部性能访问。

两臂均工程通过后，一次test270统一评估control/candidate×identity/固定八镜像。每模型八镜像第一个atom同时给identity，共2×8×270=4320病例前向，不重复identity。八镜像按既有顺序对反射还原后的raw logits作float32等权平均，BF16模型、batch8、阈值0、无后处理/Segment Anything Model/GT中心，不按test挑view/权重/epoch/model。所有四行更新榜单。

结果分层：扫描点估计超过0.757269662才称该旧cohort整体Scan超榜；identity超过0.752003176才称该旧cohort1×Scan超榜。整体稳健晋级另要求candidate Flip8高于matched control Flip8、Subject不低于旧overall .741253585、Point≥264/270，两个Dice相对旧overall患者簇95%CI下界≥0，才称本冻结CI门支持；点估计领先不冒充统计显著。identity单独报告不替代失败的整体门。所有CI含绝对与配对、10,000次簇bootstrap，seed2026091032，百分位95%，不调整多比较。数据是旧多次reused/nonblind test、非external，来源官方字节parity未知。

04:22实现前精度澄清：上述显示数值是9位小数；实际比较固定使用现有榜单/旧raw的完整精度overall Scan0.7572696616942132、Subject0.7412535853511771，1×Scan0.7520031755617648，不以显示舍入制造微小领先。旧overall报告SHA29efc557aa12b6f82fd079edd84cc3c8c1e5ad4536880c016b664932234f45ad。旧报告绝对CI曾有bootstrap种子纠正，本次不改旧CI，新的配对/绝对CI独立按上文患者簇seed计算。

## 7. 成本与风险

预计两A10040GB并行约2–2.5h/4–5GPU小时；新增checkpoint/recovery<300MiB，无新影像缓存。单模型4.70M，identity1×，八镜像8×；测两模式可复用第一atom，总4320forwards。684到864的训练分布变化可能使收益消失，无教师不保证超过旧972蒸馏。只有一个固定full seed，不称多seed完整重训确认。测试不得回流调参。

## 8. 最终结果（完成后填写）

尚未获Stage A准入、尚无full864新模型/测试成绩。仅提前准备，不宣称已启动或已支持。
