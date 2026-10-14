## 病例间并行预处理完成：两轮吞吐门通过=True

动机：约97%文件到输出时间在预处理。只将单线程CPU进程1→4，固定N4/输入/模型完整八视图/阈值/最高点/有界pending8与病例顺序；不恢复失败剪枝。

fit8扫描/4患者，两轮批量加速2.852/2.950倍；32/32输入与旧cache字节一致，同8例四次完整logit/mask SHA及点完全一致。只证固定小批量调度，不称Dice提升、单病例同比加速或新算法。

增加3个CPU进程和运行时内存；完整首例时延/批量耗时/CPU秒/采样RSS见报告。0训练/新权重/大型cache，280前向、32次N4；没有标签/开发/test/保护队列新访问。旧test270总体/1× Scan仍0.757269662/0.752003176。保留明确批量选项，不能无条件默认所有单病例请求。

[七问](archive/research_records/2026-09-11_case_parallel_preprocessing/IDEA.md)、[结果与资源代价](archive/research_records/2026-09-11_case_parallel_preprocessing/RESULT.md)、[机器结果](archive/research_records/2026-09-11_case_parallel_preprocessing/RESULT.json)、[核验](archive/research_records/2026-09-11_case_parallel_preprocessing/VERIFICATION.json)。
