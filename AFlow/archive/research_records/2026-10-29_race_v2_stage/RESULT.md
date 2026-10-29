# RACE-v2新阶段最终270评测

原270扫描／119患者，单T1ce、原生1mm原始裁剪栅格，排除填充但保留整个裁剪GT。模型与统一配方只由108选择，最终恰两个候选，无270反向调参；继承test-tuned历史，非独立盲测。

| 类别 | Scan | Subject | Point | 部署前向／例 | Scan>0.8000 |
|---|---:|---:|---:|---:|---|
| 单模型 | 0.788697343 | 0.775536380 | 251/270 | 8 | False |
| 纯SegResNet双模型 | 0.789070115 | 0.776314624 | 254/270 | 16 | False |

最终权重与统一部署配方：
- single：race164；`{"config": {"base_point_auxiliary": true, "blend": 1.0, "box_mm": 17, "dependency": "joint", "id": "race8_local_joint_k17_ridge0.2_blend1.0", "kind": "race_same_model8_local_dependency", "lattice_stride": 4, "neural_case_forwards": 8, "output_logit_threshold": 0.0, "point_rule": "final_refined_logit_argmax", "residual_clip": 2.0, "ridge": 0.2, "uncertain_logit_band": 2.0, "vote_logit_threshold": 0.0}, "id": "race8_local_joint_k17_ridge0.2_blend1.0", "mode": "race8_local_dependency", "neural_case_forwards": 8}`。
- fusion：race164 + w32e171；`{"algorithm_recipe": {"EM_iterations": 5, "background_probability_maximum": 0.01, "background_seed_minimum": 1024, "base_covariance_recipe": {"alpha_max": 0.95, "alpha_min": 0.5, "anchor": 0.75, "conflict_gate": false, "epsilon": 0.0001, "forwards_per_scan": 16, "id": "paired_cov_k5_ridge4_conflict0", "kernel": 5, "kind": "paired_covariance_shrinkage", "models": 2, "ridge": 4.0, "singular_relative_tolerance": 1e-06, "statistics_temperature": 4.0, "threshold_logit": 0.0, "zero_variance_floor": 1e-08}, "components": 1, "foreground_probability_minimum": 0.95, "foreground_seed_minimum": 64, "forwards_per_scan": 16, "histogram_bins": 256, "histogram_domain": "full_image_nonzero_min_max", "id": "appearance_k1_lambda0.5", "initialization": "equal_weights_weighted_hist_midquantiles_global_variance", "kind": "image_appearance_unary", "log_likelihood_ratio_clip": 6.0, "maximum_crop_voxels": 5591040, "mixture_weight_floor": 1e-06, "models": 2, "new_neural_forwards": 0, "range_floor": 1e-06, "sampling": "all_seed_voxels_no_subsampling", "strength": 0.5, "threshold_logit": 0.0, "trust": "one_minus_Bhattacharyya_coefficient_of_normalized_densities_on256bin_centers", "variance_floor": 0.01}, "algorithm_recipe_path": "/mlx_devbox/users/songjunyi.666/playground/AFlow/archive/research_records/2026-10-29_race_v2_inference_b/appearance_b/DEPLOYMENT_RECIPE.json", "algorithm_recipe_sha256": "b36e86bc6e6039c51d9e6d4aa1cb0f776114da7136edff2acaa87f8eb64f3a5d", "algorithm_source_sha256": "0f096a65ddc4ec4f219488f0cae447993965708f3ce9a9317a096f0e0c540332", "decoder": "raw", "primary_weight": 0.75, "space": "image_appearance_unary"}`。

训练新增80轮／9760更新，使用新AdamW，非无缝续训；训练选中best为新增0轮。新增0轮表示本次微调没有得到胜过初始化的入选权重。最终实际权重与统一配方、108选择依据和源码SHA见[冻结选择](FINAL_TEST_SELECTION.json)。

双模型相对单模型Scan差+0.000372773，患者配对95%区间[-0.00146703684181572, 0.001982819264306369]。各方法相对旧RACE完整配方的Scan／Subject配对区间、Point差和全部变差病例见[机器结果](RESULT.json)。10000患者整簇bootstrap，seed20261029；选择后描述区间未校正搜索。

原推理worker失联导致/tmp载荷丢失；原生四载荷已重建且SHA等历史，RACE按原even/odd135、batch8尾7有界重建2160，逐atom SHA等历史；W32新生成2160。新index仅语义一致、不冒充旧字节SHA。按真实完成登记4320新增病例前向，新270网络前向合计4320。两个最终配置共用相同输入的模型缓存，不重复网络。540份保存掩膜由worker本地独立重载，TP／FP／FN、Dice、定位点和SHA全核对；总控再独立核对整数统计和患者区间，未转移临床数组。

新增16项单模局部相关分歧和12项双模协方差收缩在108的均值小幅提高，但配对区间跨0，未确认稳定收益；首轮融合分折稳定收益也未确认，最终结果不抹去该限制；训练机制贡献未由同期纯模型对照单独识别。所有失败、dtype修正和冻结补记保留，临床数组向主节点备份被自动审批拒绝且未执行。

[恢复执行与原固定选择](../2026-10-29_race_v2_final270_recovery/EXECUTION_BINDING.json) · [新增推理108完整结果](INFERENCE_INNOVATION108_RESULT.md) · [单模型逐例](RAW_TEST270_single.json) · [双模型逐例](RAW_TEST270_fusion.json) · [首阶段108](INITIAL_STAGE_RESULT.md) · [训练交接](../2026-10-29_race_v2_training/HANDOFF.json) · [执行边界](RUNBOOK.md)
