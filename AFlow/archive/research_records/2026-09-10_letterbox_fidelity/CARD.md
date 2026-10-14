## 保长宽比补边：fit标签保真诊断 ✅ 完成，未准入训练

**一句话：** 保住照片形状会减少短轴像素，已量化这项取舍。

**出发点：** 原脑裁剪三轴尺度差中位1.2879；旧标签往返0.965432不是模型上限。

**具体做法：** 唯一变化为最长轴128、短轴同比取整/居中补边；原生WT、image-only裁剪和最近邻前后映射不变，两个方案回相同完整原生标签评分。

**结果：** fit684/313标签Scan往返Direct=0.965432464，补边=0.937513853，差-0.027918611；Subject分别0.963994738/0.935851554。补边缺失组件310（原184）；新增>125体素缺失组件0，整WT空0，中位补边比例36.10%。未运行模型/开发，无置信区间。

**是否支持直觉：** 支持可以保近似长宽比的几何事实，但标签保真差-0.027919揭示分辨率代价，不证明模型会升或降。固定工程否决门=False；即使不否决也不自动批准训练。原crop外4体素保留记录、不排例。

**代价/下一步：** 55.0秒/4 CPU，0 GPU/模型前向、无新像素缓存。暂不挤占旋转四模型；新训练仍须共同网格评测与独立预算。[七问](archive/research_records/2026-09-10_letterbox_fidelity/IDEA.md)、[结果](archive/research_records/2026-09-10_letterbox_fidelity/RESULT.json)、[原始逐例](archive/research_records/2026-09-10_letterbox_fidelity/RAW_CASES.jsonl)、[限定核验](archive/research_records/2026-09-10_letterbox_fidelity/VERIFICATION.json)。
