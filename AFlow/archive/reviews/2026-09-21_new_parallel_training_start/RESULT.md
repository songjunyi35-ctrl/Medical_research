# 新批次并行执行启动核验

组合组双卡烟测通过，原始缓存正在CPU构建，GPU正式训练尚待缓存及固定参考验证门。患者等权单卡组已通过影像缓存SHA、患者损失权重及合成GPU检查，正式训练已完成至少1轮，972扫描/444患者完整覆盖，每轮权重和约972，未按batch重新归一。两组各自冻结与监督，不增加seed，不做当前test调优或自动270评估。

[机器快照](RESULT.json) · [组合实时阶段](../../research_records/2026-09-21_segresnet_pretrain_native_res/PIPELINE_STATUS.json) · [患者等权实时阶段](../../research_records/2026-09-21_segresnet_patient_equal_loss/PIPELINE_STATUS.json)。
