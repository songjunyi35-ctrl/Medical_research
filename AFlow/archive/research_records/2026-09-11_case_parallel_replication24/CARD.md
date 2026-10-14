## 24扫描不同fit患者批量复制完成：固定门通过=True

动机：旧8例工程探针不足以保证其他工作负载；排除旧4患者，按清单取24扫描/12患者。只比较CPU进程1→4，单线程N4/pending8/固定native模型八视图/输入阈值与点不变，旧SHA绑定代码保留。

两轮批量加速3.516/3.517倍；96/96输入byte exact，24病例四次完整logit/mask/point一致。支持更大批量下的工程复制，不称单病例延迟同比改善、CPU量减少或精度提升。首例时延与CPU/RSS代价完整报告。

0训练/新checkpoint/大cache，792模型前向、96次N4；仅fit图像，无标签/开发/test/保护队列访问，旧test270总体/1× Scan仍0.757269662/0.752003176。两个不同fit队列不是外部临床验证；不按结果改进程网格。

[七问](archive/research_records/2026-09-11_case_parallel_replication24/IDEA.md)、[完整结果](archive/research_records/2026-09-11_case_parallel_replication24/RESULT.md)、[机器结果](archive/research_records/2026-09-11_case_parallel_replication24/RESULT.json)、[独立账目核验](archive/research_records/2026-09-11_case_parallel_replication24/VERIFICATION.json)。
