# Implementation amendment 01: enforce first-case-before-full-cache order

状态：双源下载完成但190个NIfTI均未解码、label values未访问、cache/model不存在。

预注册要求固定首例先解码并通过，再打开其余94例。初版cache脚本虽有首例独立重建，却会先并行打开95例，顺序不够严格。修订新增`probe_brats_africa95_first_case.py`，只读取冻结顺序第一例并保存geometry/label-domain/128³ hash；全量builder强制验证该收据，并要求全量首例输出hash与probe exact。

病例、文件、all-positive WT、N4/crop/128³、模型、阈值、视图、指标、bootstrap和停止门均不变。该加固发生在第一次NIfTI/label-value访问前。
