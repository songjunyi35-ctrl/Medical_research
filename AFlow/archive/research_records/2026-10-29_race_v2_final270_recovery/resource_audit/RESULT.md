# 1154259训练终态后的独立资源审计

2026-10-29 19:16:43北京时间开始，19:17:36完成；状态passed。此审计不分配GPU、不启动训练或推理，没有模型前向或GPU张量分配，未访问270标签。

- 原协调器/launcher/torchrun/双rank PID 2941、4603、4604、4608、4609均已退出。审计开始时无GPU compute进程，两卡均0 MiB、0%。
- GPU0 UUID为`GPU-ea400707-0b2b-1da8-ef21-5e6f77aaa52e`；GPU1为`GPU-e7e4ead3-6461-4fac-ac02-2e1e666fa462`，均A100-SXM4-40GB。CUDA可用，torch检测两卡。
- 环境Python3.11.2、torch2.7.1+cu126/CUDA12.6、MONAI1.5.2、nibabel5.4.2、SimpleITK2.5.5；驱动535.161.08。
- ROOT空闲24,652,021,760 bytes（约22.96GiB），worker/tmp空闲4,274,550,468,608 bytes。未删除或转移任何旧数据。
- 本worker自己的972训练和108验证缓存均保留；两个index SHA相符，9个实际payload共45,254,246,400 bytes全部重新逐字节SHA，与原DATA_train/DATA_validation记录一致。未将1154121缓存当成本机缓存。
- 原EXECUTION_FREEZE的25份源码、recipe/budget/IDEA和数值函数AST均通过验证，原80轮/9760更新终态和best新增0轮不变。
- 持久portable best/latest及独立fixed_e000 SHA一致；portable best/latest和主节点/tmp完整best/latest恢复包均CPU strict重载通过，恢复包优化器计数通过。worker run与固定backup inode路径的四份完整权重均重新SHA，与既有恢复收据相同。

证据为[WORKER_RESOURCE.json](WORKER_RESOURCE.json)、[LOCAL_TRAINING_FREEZE.json](LOCAL_TRAINING_FREEZE.json)及[SUMMARY.json](SUMMARY.json)；可复查审计入口[audit.py](audit.py)。原训练目录、代码、配方、缓存和权重均未写入。新恢复授权与队列由总控单独管理。

1154121失联原因未知；本审计没有证据认定是利用率查杀。
