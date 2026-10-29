# RACE164固定270评测复现与边界

本轮已完成后不重启。当前入口为`scripts.race_inference_test270_20261029.coordinator_v2`，两个独立GPU进程运行`evaluate_v2 --rank 0/1`，统计与出版入口`publish_v2`。旧`coordinator/evaluate`因为缺少通道维在第一个预测形状断言失败，实际神经前向0；旧源码和访问收据保持历史，补充冻结为EXECUTION_FREEZE_V2。不是重复完成的患者前向。

固定模型为`workspace/BraTS/segresnet_t1ce_wt/race_20261028/race/best.pth`，第164轮，SHA见MODEL_PREACCESS/COMPLETION_AUDIT。预测器接口在predictor.py：`forward_view(model,image,clean_geometries,flip)`只接`[B,1,160,208,168]`图像和图像几何；`decode_outputs(aligned_logits,image,config)`只接自身预测、图像及已冻结参数。标签与病例ID均由外部保存/评分代码处理，不输入API。

worker1154121双A10040GB，缓存和每例8个原始float32 logit均只存worker的`/tmp/aflow_race_e164_test270_20261029`。该/tmp与主机/tmp不同；短期节点退役后不可声称这些载荷仍可直接复制。原始MRI/NIfTI与训练权重未删除，建设脚本和历史载荷SHA保留。新用户复现需要对应数据及权重，Git不提供这些私有/大载荷。

准备阶段cache.py的测试路径字段错误和cache_v2.py的canonical标签负步幅错误保留；实际成功为cache_v3.py，复用1例完整载荷，补齐269例。不能据此声称整个失败阶段每例只执行一次N4。CACHE_READY四SHA完全匹配历史，index增加图像反射矩阵后有新SHA。大数组只worker/tmp，持久预测为两个输出共540份压缩位掩膜。

PREDICTOR_FREEZE固定模型接口及原推理模块；EXECUTION_FREEZE固定初版入口/参数/七问；EXECUTION_FREEZE_V2额外固定通道维修订入口。不得覆盖这些源码或修改SELECTED_RECIPE来复用现有成绩。MODEL_PREACCESS和每rank访问收据独占保护，真实执行的8×270前向由FORWARD_COMPLETE与原始atom SHA清单记录，原始identity共享，无第9次前向。

若仅出版/统计失败而raw完整，可在原节点以`evaluate_v2 --rank N --statistics-only`只从既存8视图raw解码，无神经前向；预测已存在时须逐像素/点一致，不静默覆盖。原子文件不全时不得据此重复网络。当前完整结果不需要也不授权再跑这些恢复入口。

RESULT/RAW_TEST270_raw/RAW_TEST270_enhanced保存完整逐例数据；INDEPENDENT_MASK_AUDIT核对540掩膜；COMPARISON_CONTEXT说明旧最佳Point下降4例和总体nnU-Net仍更高。当前正差是同一复用内部270队列上的描述性证据，已有旧test选参历史。没有当前test参数搜索、新训练或outer/confirmation。
