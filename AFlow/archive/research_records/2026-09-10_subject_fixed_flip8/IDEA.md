# Experiment idea: 对数据采样较强的同一模型固定八镜像平均

Status: preregistered_before_any_new_flip8_member_prediction

## 1. 现在的问题是什么？

患者均衡两个模型dev Scan0.753475/0.755906、Subject0.740633/0.742725、Point167/168；其1×原配方仍因seed11 Point门关闭。与占比模型等权融合只让seed11 Scan+.000642、seed12 Scan−.005457，未稳定超过强成员。因此不再稀释强模型，检验同一权重不同观察方向。

## 2. 一句话直觉

让同一个模型从镜子里的八个方向看同一幅图，把还原方向后的意见平均；不换模型，也不混较弱模型。

## 3. 为什么现在选？

项目9月4固定Flip8在两个内部复制上平均Scan+.006828，且同checkpoint旧test上+.005510；是本项目已有推理依据，不是保证当前患者采样权重收益。该历史结果含已知teacher exposure限制，不能借作新独立确认。此处两模型teacher-free、只fit684。旧1×采样配方和刚结束跨配方融合保持关闭；新方法只检验固定多视图推理，区别于训练形变/标签变更和多模型平均。

## 4. 具体做法

固定subject_balanced seed2026090711/12、epoch168→原128³影像→全部八个空间轴镜像组合→每次同一模型预测→反转回原方向→按历史顺序float32 raw-logit等权平均→阈值0/argmax→原hard GT评分。Identity复用首atom。没有训练、GT路由、视图选择、概率平均或后处理。

## 5. 唯一变化与固定项

唯一变化是1×到固定8×。checkpoint、fit684/313、dev180/78、图像/GT/cache、batch8/BF16、阈值0、原顺序/等权公式不变。两个seed都验证，不按dev挑权重/seed。旧1×结果完整逐例parity后才接受新结果。

## 6. 成功与停止

采用既有Flip8方法的方向/CI安全框架（不是放宽其他训练方法的门）：每seed Flip8−Identity Scan>0、Subject>0、Point≥−1，两Dice10k患者簇95%CI下界≥−.003；本次进一步要求两seed平均Scan增益≥.003、Point分别≥169/162（恢复相对native170/163最多损1的底线）。bootstrap seeds2026091121/22，百分位CI，未多比较校正。任一失败关闭此固定8视图应用，不搜索镜像子集/权重/阈值。

若全门过且有时间，事前固定seed11允许单一test270面板（same checkpoint identity/Flip8两输出，2160forwards），均入两榜，成本1×/8×。整个8小时仍最多一个test panel，此前没有。旧test非盲非外部/历史test-informed。test前另冻结checkpoint/data/code/output/bootstrap和旧榜目标，时间不足则不测。

## 7. 代价与风险

训练0；开发180×2×8=2880模型forwards，预计几分钟；fit8 smoke128forwards。部署同4.70M权重但8×推理，比2模型16×便宜、不等于1×。镜像平均可能平滑真小灶或不能恢复定位，任何过门结论仅限复用开发。当前剩余不足1h，不启动新训练或无界搜索，截止11:28:19 UTC。

## 8. 当前结果

未实施/读本固定患者模型的8镜像结果。完整证据来源：[既有Flip8 IDEA](../2026-09-04_direct_locator_flip8_tta_stability/IDEA.md)、[既有协议](../2026-09-04_direct_locator_flip8_tta_stability/PREREG.md)。
