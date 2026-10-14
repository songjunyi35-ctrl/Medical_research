# 训练标签引导的每病例最佳老师蒸馏结果

## 结论

GT（ground truth，真实标签）引导的每病例最佳teacher routing在Full972训练中确实持续选择了三位不同teacher，但fixed validation108上Candidate的Scan、Subject和Point全部低于matched Control与archived Full972 seed32。Promotion全面失败，不访问test270，也不调整routing、teacher、seed、KD权重、epoch或阈值。

Candidate Scan/Subject/Point=`0.732910/0.725998/100/108`；matched anchor-repeat3 Control=`0.736013/0.728973/103/108`；archived Full972 seed32=`0.740013/0.731266/103/108`。

通俗地说：训练时“每道题听当下答得最准的老师”看起来合理，三位老师也都实际被选中；但student面对不断切换的teacher函数后，到了validation反而不如始终听同一位anchor。机制执行成功，泛化主张失败。

## 方法与唯一变化

- 两臂均在Full972 972 scans / 444 subjects上，以相同student seed20261040、exact initial state、sampler indices、增强tensor和固定epoch168训练。
- 模型均为4,700,897参数plain SegResNet；真实标签loss均为Dice+0.1 CE，蒸馏均为weight0.3、temperature2的binary voxel KL。
- 每步两臂都恰执行3次frozen teacher forward。
- Control把Full972 seed20260932 teacher真实forward3次，并使用相同anchor raw logit。
- Candidate让Full972 seed20260932/40/41各forward1次，对每病例以GT soft-Dice loss argmin选择整幅teacher raw logit。
- 阈值0、Point为raw-logit argmax、无TTA/SAM/后处理；部署均为1×student。

## 实现与工程过程

- Per-case soft-Dice、argmin/gather、tie first-index、detach与错误输入4项CPU合同通过。
- Paired真实GPU smoke：initial state、sampler、首批增强image/label exact；前16 draws Candidate routing=`5/4/7`；teacher forward与routing计数合同通过。
- Formal双臂均完整168轮、每轮121步/968 draws、168个sampler SHA exact、checkpoint reload exact、全部工程门通过。
- 用户中途消息中断parent launcher，因此没有`FORMAL_EXIT.json`；两个已运行的child训练进程未中断，独立完成并写出权威`TRAIN_DONE.json`和checkpoint。该边界在validation pre-access receipt中冻结。

## Routing分布

Candidate全168轮共162,624个训练draw：

- Full972 seed20260932：47,215次（29.03%）
- Full972 seed20260940：63,869次（39.27%）
- Full972 seed20260941：51,540次（31.69%）

Routing没有塌到单teacher，seed40被选最多。这证明新机制确实工作；失败不能归因于router没接上或teacher未被使用。

## Fixed validation108 / 50 subjects

| 方法 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| Archived Full972 seed32 | **`0.740012535`** | **`0.731266217`** | **`103/108`** |
| Matched anchor-repeat3 Control | `0.736013432` | `0.728973188` | **`103/108`** |
| GT-best-of3 Candidate | `0.732909616` | `0.725997970` | `100/108` |

Candidate相对Control：

- Scan `-0.003103817`，95% CI `[-0.010461848,+0.003822867]`；53/108 scans改善、55变差。
- Subject `-0.002975218`，95% CI `[-0.011860025,+0.005455025]`；24/50 subjects改善、26变差。
- Point`-3/108`。

Candidate相对archived Full972 seed32：

- Scan `-0.007102919`，95% CI `[-0.016251576,+0.000924337]`；54/108 scans改善、54变差。
- Subject `-0.005268247`，95% CI `[-0.015250127,+0.004309784]`；25/50 subjects改善、25变差。
- Point`-3/108`。

## Promotion与解释

所有九项promotion门失败：Candidate没有超过Control/archived的Scan或Subject，Point低于103，四个CI下界也低于`-0.003`。

最可信解释是：GT hard routing给每个病例提供了局部更好的teacher target，但不同病例/epoch之间target函数不连续，同容量student需要同时拟合三种seed偏好，反而损伤一致的泛化表示。训练teacher oracle质量不等于student可学习、可泛化的知识。

当前结果只否定固定三个teacher、per-case hard argmin、output-only voxel-KD的实现。它不证明所有adaptive multi-teacher方法无效；但本validation已揭示，不得事后改成soft weighting、inverse loss、entropy gate或per-voxel routing并在同一cohort验证。

## 成本

- Control/Candidate formal：`14,883.58s / 14,861.63s`，各约4.13h，两卡并行墙钟约4.13h。
- Peak allocated：`20.309/20.372GiB`。
- 复用现有cache与teacher checkpoint；新增两个student checkpoint和报告，无大cache。
- 推理仍4.70M、1×，但Candidate未晋级。

## Evidence与访问

- Raw validation SHA-256 `ce89d554a792fe627bdf86bd3a7e4673b7efc754c9962404972b62a6427ce5ae`。
- Validation started SHA-256 `c47bb277496c5de090aa4e574e6c40dd07b850c80e59c60b614bd87c124c8624`。
- Validation pre-access receipt SHA-256 `9d08d647b1cf7f90a10a83bbaeafffb3c384417051267e3eadc2e5a82f8873d5`。
- Control/Candidate TRAIN_DONE SHA-256 `6fb33ec8810f8cd6622c30ff1e8796517a519d4f5b158d772df2da0ff2984594` / `2b0635886298ffbf30c7710dfff3e0fbc029fc6de9a4cdf3b28b68c6eb7b97da`。
- Control/Candidate checkpoint SHA-256 `a128ba15906fb55824f74828e9d0566c01d3389aaa6f92baee7ab7f6d4b06be7` / `75c8f669a8b25ab92f4973a7e4f708f8cf1e3d4fe650393c36644bcdc8492473`。
- Fixed validation108已访问；outer、confirmation、test270均未访问。

## 下一决定

关闭GT-best teacher routing分支，不进test、不调routing。连续两次multi-seed实验说明：seed多样性可以救弱轨迹，但固定平均和GT hard routing都不能稳定超过强单模。下一步科研应转离“如何组合这三个seed”，优先寻找独立外部cohort或新的数据表示/训练信息，而不是继续在已揭示validation上设计teacher权重。

