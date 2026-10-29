# B最终270固定融合入口（尚未启动）

只由root真实`FINAL_TEST_SELECTION.json`触发，schema文件不是授权。先验证完整训练终态/预算、108审计、<=2候选、数据/源码/模型SHA；B只运行root选中的融合，不搜索270。

模型READY统一接口：checkpoint/manifest/index SHA、canvas、2160唯一slot/view及每atom SHA/shape。RACE164或新RACE全部由A生成/复用，B只读；w32本节点270缓存实际缺失，只有最终融合选中w32时才一次2160前向。记录独占PREACCESS，不自动神经重试。

独立进程依次运行，避免GPU锁跨阶段重入：

1. `python -m scripts.race_v2_inference_b_20261029.final270 synthetic`：只全尺寸纯合成门，无270像素/评分。
2. root最终选择包含w32时：`python -m scripts.race_v2_inference_b_20261029.final270 forward-w32`，GPU1生成固定e171八镜像。270画布从已核验native index读取，尾批按选择batch实测。
3. A两个模型READY齐全后：`python -m scripts.race_v2_inference_b_20261029.final270 score --candidate-id ROOT_SELECTED_ID`。仅固定统一权重/域/解码；预测函数不接GT或病例ID。全270 GT仅最终scorer，所有掩膜和点、TP/FP/FN、SHA保留供root配对区间审计。

命令环境：worker1154121、CUDA_VISIBLE_DEVICES=1、PYTHONPATH为工作区；数组/掩膜仅worker/tmp，文本在B/final270。历史test-tuned披露保持。
