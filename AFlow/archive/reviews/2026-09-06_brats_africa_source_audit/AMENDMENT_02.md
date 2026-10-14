# Amendment 02: lower directory concurrency

首次bounded-retry执行列到146个病例目录后，8并发列病例文件时有一条路径连续3次TLS `WRONG_VERSION_NUMBER`。仍无inventory/report/datainfo归档、无payload或数组访问。

唯一修改：最大目录并发由8降为4；每页最多3次、0.5/1.0秒短退避保持不变。该值与同一Faspex站点先前UTSW全量目录审计成功配置一致。若本次仍有TLS/HTTP失败，不再继续调并发或无限重试，本轮来源门停止。所有科学/文件/访问门不变。
