# 自动模型固定工作流：工程闭环结果

状态：工程执行完成；不是工作流搜索完整科研闭环，也不是精度晋级。


原预注册七问保持不变。仅改变视图集合；无训练、标签、开发/test/outer/confirmation/外部性能访问。

Seed 2026090711：1×/2×/4×相对full8配对中位加速为7.818倍/3.962倍/1.993倍。

Seed 2026090712：1×/2×/4×相对full8配对中位加速为7.829倍/3.965倍/1.994倍。

16项GPU子集parity精确一致，实际1654病例前向；工程速度门通过=True。当前硬件推理阶段74.5秒，不包含预处理/文件加载，不是端到端加速。无新checkpoint或大型cache。支持按需执行确实省计算，不证明剪枝精度安全。下一步另冻精度约束搜索协议，尚未准入测试。

[七问](archive/research_records/2026-09-11_automatic_workflow_engineering/IDEA.md)、[完整结果](archive/research_records/2026-09-11_automatic_workflow_engineering/RESULT.md)、[机器结果](archive/research_records/2026-09-11_automatic_workflow_engineering/RESULT.json)、[验证](archive/research_records/2026-09-11_automatic_workflow_engineering/VERIFICATION.json)。
