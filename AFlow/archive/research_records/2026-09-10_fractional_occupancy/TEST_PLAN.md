# 条件性固定seed11测试细则（占比模型新dev/test前）

状态：预先准备，尚未准入。遵循IDEA/PREREG既有单panel条件，不改变训练或开发门。

1. 问题：旧test270/119整体Scan0.7572696616942132、1×Scan0.7520031755617648；占比模型尚在训练、无新dev结果，不能用开发分数代替榜单成绩。
2. 直觉：开发双seed确认训练办法后，用事前固定seed11做一次同旧榜考试，不挑成绩更好的seed。
3. 原因：本方法试验只改变训练标签表示；旧弱形变已失败，所以本条件分支合法接续。共享测试时八镜像（TTA）仅复用预先指定的旧有效推理办法，不按新test选视图。
4. 流程：两个source checkpoint（旧native control11、新occupancy11）→同128³缓存→原输出及全部8个空间轴镜像还原raw logits等权float32均值→logit>0与argmax→原硬GT评分。batch8、BF16模型、无后处理/SAM。identity复用首atom，总4320病例forward。
5. 固定项：seed11、epoch168、仅fit684/313、checkpoint/训练报告/数据和代码SHA在preaccess绑定；seed12仅作为开发复现，不按dev择优；旧dev180仍非训练。test270固定manifest/index/像素SHA沿用公开项目旧freeze，非新增外部/盲测。
6. 成功/停止：只有两seed开发全部门过、独立验证通过、全session没有其他test panel且至少10min测试/归档余时才准入。candidate原输出超过0.7520031755617648为此cohort1×Scan点估计超榜；candidate八镜像超过0.7572696616942132为整体Scan点估计超榜。更严格的完整科学门另外要求八镜像高于同checkpoint种子的control八镜像、Subject≥0.7412535853511771、Point≥264，candidate−旧overall两Dice患者簇95%CI下界≥0。1×门独立，不作为整体完整门的附加条件。统一10,000次簇bootstrap、seed2026091032、百分位95%、未多比较校正。负/平/失败门和control全部四行更新两榜；不重复或调参。
7. 代价：0额外训练，4320forwards；保存四份二值mask共270MiB用于独立Dice核验，原始logits不存；不独立重算argmax，只核保存点坐标GT命中。输出RAW_TEST270/TEST_MASK_INDEX在本记录，独立测试归档写到相邻2026-09-10_fractional_occupancy_test270/RESULT.md/JSON，保留本目录开发RESULT不覆盖。代码/输出冻结后才读取新测试预测。

本项目历史test结果已知、基线流水线test-informed；“本新panel不调参”不是“整个研究从未受历史测试影响”。所有来源限制保留，不称外部泛化或临床可用。
