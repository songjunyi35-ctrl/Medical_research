# Experiment idea: Full972更宽SegResNet完整voxel-KD轨迹

Status: planned and frozen before implementation, model construction, or new validation access.

## 1. 现在的问题是什么？

Full972 width16 seed32在reused internal test270达到`0.751760/0.737327/262/270`，但final train Dice `0.7784`低于matched Full756的`0.7988`。新增216例与同计算量带来更广泛分布，4.70M模型可能欠容量。

## 2. 一句话直觉

教材变厚后，除了增加学习时间，也可能需要更大的“脑容量”才能把多样病例学完整。

## 3. 为什么选择这个idea？

- 欠拟合迹象来自train-only曲线，不依赖已揭示test选择设置。
- Full972的数据和完整轨迹机制已经相对matched control成立。
- 160³失败改变的是输入网格和近2×体素计算；DMRFB失败是late adapter。二者均未测试随机初始化的全程宽度扩展。
- Width24是单个预注册容量点，不搜索16/20/24/32网格。

## 4. 具体怎么做？

阶段A：在同一candidate-fit864/common-inner108上，control使用`init_filters=16`、4,700,897参数；candidate使用`24`、10,574,161参数。两臂seed20260960、Full972病例角色、968 draws/121 steps、history/EMA teacher、DiceCE+voxel-KL和200轮schedule相同。Width24在post100按Scan→Subject→早epoch选择唯一epoch，和width16同epoch比较。

阶段B：只有inner门通过，才用新seed20260961在full972上分别从随机初始化训练width16/24到同一个selected epoch，然后一次性报告fixed validation108。

## 5. 唯一变化与固定项

- 唯一变化：`init_filters 16→24`，参数`4,700,897→10,574,161`。
- 固定：Full972数据/角色、common inner、随机seed、draw/step、teacher、loss、LR、增强、epoch选择、threshold0、无TTA/SAM/postprocess。
- 部署均是单模型1×forward；width24参数与计算更高但不是ensemble。

## 6. 成功、失败和停止规则

- Inner candidate只在epoch101–200按Scan→Subject→早epoch选择。
- Inner必须严格超过width16同epochScan/Subject；Point最多差1/108；Scan/Subject paired 95% CI下界>=-0.003。
- Final width24必须严格超过matched width16和现有Full972 width16 seed32 validation Scan/Subject，Point>=101，四CI安全门通过。
- 任一NaN/Inf、teacher gradient、initial seed/step/hash错误、peak allocated>32GiB、单臂>8h、checkpoint state不exact立即停止。
- 研究发生在test揭示后；test不能选择宽度、epoch、loss、阈值或后处理。

## 7. 成本与风险

- Inner两卡并行预计width16约4.5h、width24约5.5–6.5h；若过门，final再约4–6h。
- Width24显存预计25–30GiB，硬门32GiB。
- 新增checkpoint/report预计<1GB；当前磁盘16GB足够，不建新cache。
- Width24推理仍1×，参数约2.25×；风险是更宽模型过拟合或teacher限制学生上限。

## 8. 最终结果（完成后填写）

- Inner：待运行。
- Final validation：仅inner过门后允许。
- Post-test evaluation：仅新预注册与明确授权后允许。
