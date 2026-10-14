## 小角度旋转全864配对重训 ⛔ 主开发门失败，未准入

**一句话：** 只有旋转两个种子共同可靠，才全864重训；本次前提未成立。

**出发点：** 旧test270/119 overall Scan0.757269662、1×Scan0.752003176，开发高分不能直接当超榜。

**具体做法：** 原计划全864同采样control/rotation、固定seed2026091071、168轮，唯一旋转差异；随后一次四输出同test270。此计划没有正式执行。

**结果：** 主旋转两个采样层都未过联合门，入选层为空。本分支0训练、0test、无新checkpoint；原dev180没有成为训练。不是“全864模型试过并失败”。

**是否支持直觉：** 本分支未检验。主开发具体旋转配方已判负，不泛化为所有空间增强或全数据重训不可能。

**代价/下一步：** 仅准备代码和864元数据清单，未产生全数据GPU训练成本。关闭本分支；保持fit684/313与reused dev180/78，另行执行事前固定的图像模糊备用。[未准入结果](archive/research_records/2026-09-10_small_rotation_full864/RESULT.json)、[主旋转结果](archive/research_records/2026-09-10_small_rotation/RESULT.json)、[七问](archive/research_records/2026-09-10_small_rotation_full864/IDEA.md)。
