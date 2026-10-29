# 对侧图像证据108与最终入口交接

- 本轮8新配置已完成，108/50原生1mm、2纯SegResNet/16前向成本；新增网络0、无270访问。RESULT.best_new固定一条contra_absolute_k11_b0.2；包含旧参考时既有cov仍最高，不强迫新方法入选。
- RESULT/RAW/DEPLOYMENT_RECIPE与EXECUTION_FREEZE已冻结。RAW有11×108完整计数、最终score Point、worker/tmp压缩掩膜SHA；REGIONAL_DIAGNOSTIC_SUMMARY/逐例区域诊断只在scorer看GT。三个冻结控制逐例所有原始计数/点exact。
- 新代码 scripts/race_v2_inference_b_20261029/image_conflict_b/。deploy(logits_primary,logits_partner,image,recipe)仅接受两组8视图与原图，无GT、ID或查表。fit_plane在去填充crop重拟合，图像与有效支撑同反射，平移共轭门已过。
- final_snapshot/RESULT.json、RAW_VALIDATION.json等完整原80训练终态+base8view+旧fusion/cov转移完成，再仅迁移单一配方，不重开8网格。best0显式NO_NEW_MODEL skip，不重复RACE164前向。transfer waiter15135。
- 新final dispatcher watch15136替换旧空闲11777；cov transfer11776和snapshot9260保留。B/FINAL270_CONTRALATERAL_ENTRY_FREEZE.json flat源供root最终selector绑定，支持legacy/cov/contra融合，允许A独立_healthnorm single同selection共存，但fusion只base模型。root真正FINAL_TEST_SELECTION才触发，不读schema。w32如必要仅一次2160forward；RACE复用A。无自动网络重试。
- 新final固定score后独立读取全部270保存掩膜重新计数+point，audit齐全才发布B/final270/RAW_FUSION.json。旧source、冻结与结果原样保留；训练worker/进程/配方未动。
- 医学强度差是代理，无T1前后对照不称实证强化。历史108与test-informed路线、旧锚点选择全部披露；五折是条件稳定性，不是独立泛化确认。
