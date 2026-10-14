## 文件读取到输出总耗时检查完成：阶段加速不能冒充总加速

动机：N4等预处理可能主导总时长。fit前4图像、两模型两策略16次完整流程；仅改镜像集合，16/16重建输入与旧cache字节exact，其他因素不变。

Seed 2026090711：full8/pruned平均总秒12.2854/11.9736，配对总加速中位1.0266倍，full8预处理占96.97%。

Seed 2026090712：full8/pruned平均总秒12.2367/11.9976，配对总加速中位1.0224倍，full8预处理占96.97%。

计时为暖页文件读取到128³CPU输出，不含模型加载/原生导出/临床其他步骤；四例无群体CI。不能拿模型阶段3.96/2.65倍替代这里。0训练/标签/开发或测试性能访问，147前向、16次N4，无新大cache/权重；不据此改已结束方法。

[七问](archive/research_records/2026-09-11_workflow_file_to_output_timing/IDEA.md)、[结果](archive/research_records/2026-09-11_workflow_file_to_output_timing/RESULT.md)、[原始计时](archive/research_records/2026-09-11_workflow_file_to_output_timing/RAW_TIMING.json)、[核验](archive/research_records/2026-09-11_workflow_file_to_output_timing/VERIFICATION.json)。
