# UTSW-Glioma 来源审计 Amendment 02：仅传输稳定性

冻结时间：2026-09-05T19:04:53Z，第二次完整遍历之前。

Amendment 01 的首次执行已在公开 OAuth、根目录和625病例分页完成之后，于并发病例目录 POST 遇到 TLS `WRONG_VERSION_NUMBER` 并中止。没有结果/清单文件、像素、标签或模型访问；失败收据见 `EXECUTION_01_FAILURE.json`。

本修订只改变网络传输稳定性：

- 并发目录请求从12降到最多4；
- GET/POST 的连接/读取错误最多重试3次，0.5秒指数退避；
- 不重试或忽略任何数据门、HTTP永久错误或内容不一致。

625/362 cohort、6349文件、精确9/11文件集合、symbolic-link/正大小、0像素/0模型门全部不变。Amendment 01 原脚本保留，第二次执行通过独立 wrapper 注入有限重试和并发上限。

