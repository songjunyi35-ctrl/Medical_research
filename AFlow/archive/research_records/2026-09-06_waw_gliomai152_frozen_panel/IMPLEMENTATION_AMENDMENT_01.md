# Implementation amendment 01：预访问字段路径

第一次正式命令在`p.docs()`与读取cache收据之后、CUDA device检查、started receipt、checkpoint加载和任何model forward之前失败。原因只是`MODEL_PREACCESS.json`把评测器SHA存为`code.evaluator_sha256`，评测器误读顶层`evaluator_sha256`。

允许且唯一的改动：评测器读取`preaccess["code"]["evaluator_sha256"]`；不改变152病例、图像/cache、模型、view、阈值、指标、bootstrap seed、解释门或一次性规则。旧收据与失败记录保留，使用新评测器SHA另冻`MODEL_PREACCESS_V2.json`后才允许再次启动。0 checkpoint、0 forward、0性能访问，因此这不是结果驱动重跑。
