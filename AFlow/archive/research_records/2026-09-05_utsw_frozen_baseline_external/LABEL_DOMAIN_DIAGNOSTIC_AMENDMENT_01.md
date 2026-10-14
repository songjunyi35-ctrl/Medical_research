# 标签域诊断Amendment 01：可恢复进度与异常收据

首个诊断进程终止但因观察session号未保留而没有可恢复stderr，也没有结果文件。它不能解释标签域。

本修订不改变362病例、unique/count统计或解释门，只增加：每25例写原子progress；逐例显式释放数组；任何异常写FAILURE_V2含case/traceback；成功输出独立`LABEL_DOMAIN_DIAGNOSTIC_AMENDMENT_01.json`。0图像/N4/cache/model/指标不变。

