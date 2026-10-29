# 固定w32第171轮无训练推理研究：最终报告

本研究持续12.021小时，达到至少12小时窗口。固定单T1ce、SegResNet32通道第171轮及原972/108/270角色；270测试扫描/119患者、原native1mm去填充栅格。新网络训练0，不使用其他模型权重，不新增108/outer/confirmation评测，不干预独立RACE。用户授权直接用270反馈全局选择，**全部结果test-tuned，未独立确认泛化**。

| 同权重方案 | Scan Dice | Subject Dice | Point | NN前向/例 |
|---|---:|---:|---:|---:|
| 原始推理 | 0.773228040 | 0.760026920 | 256/270 | 1 |
| 研究前固定五镜像/体积解码 | 0.780995198 | 0.768306279 | 255/270 | 5 |
| 七视图成本替代 | 0.784072660 | 0.771075667 | 260/270 | 7 |
| 本轮最高 `0036_local_brain_joint` | **0.784382113** | **0.771378811** | **260/270** | **8** |
| 高成本增强融合 | 0.784354741 | 0.771605680 | 259/270 | 80 |

对原始推理差 **+0.011154073**，患者配对描述性95%区间[0.007136013411633419, 0.01651094766791356]；对研究前五镜像差+0.003386915，区间[0.0014956890700031417, 0.005926036315733135]。这些区间在大规模全局选择之后计算，未作多重搜索校正，不能当独立确认或稳定泛化保证。Scan是逐扫描均值，Subject是119患者各自均值后再平均，Point按原定义另列。

## 固定配方与部署核验

固定全局配置：`{"aggregation": {"id": "base2_brainmargin0", "kind": "brain_domain", "base": {"id": "case_agree_t0.2_w24.0_k8_logit_T4.0", "kind": "case_consistency", "vote_threshold": 0.2, "exponent": 24.0, "keep_views": 8, "representation": "logit", "view_temperature": 4.0, "views": [0, 1, 2, 3, 4, 5, 6, 7], "neural_case_forwards": 8}, "margin": 0, "views": [0, 1, 2, 3, 4, 5, 6, 7], "neural_case_forwards": 8}, "decoder": {"post_temperature": 1.125, "bias": 0.375, "beta": 0.5, "volume_scale": 1.0}, "postprocessing": [{"id": "c3_min512_protectFalse", "connectivity": 3, "kind": "minimum", "value": 512, "protect_largest": false}, {"id": "fill_small_hole1000", "source": "selected", "operation": "fill_holes", "value": 1000}]}`。概率按65536级量化，使用只读预测总直方图的体积解码；不存在逐例GT最优阈值选择。

聚合`brain_domain`，解码`brain_domain_volume`；完整全局参数见[便携配方](PORTABLE_RECIPE_CURRENT.json)和[配方链](candidates/0036_local_brain_joint/RECIPE.json)。部署只接图像和image-derived去填充几何，不输入GT、病例ID或270个逐例切点查表。图像脑区约束仅限制预测与无标签计数，评分始终保留整个原native crop GT，支撑外阳性仍计FN；原历史裁剪范围没有改变。

全部270选中掩膜/TP-FP-FN/点由原缓存概率或logits独立重建并与搜索矩阵对齐。独立API重新神经前向范围为**完整270例**，mask差0体素、点相同、逐例Dice差0，训练0。此前所有输入增强已真实覆盖270，这里的fresh是重复工程核对，非新队列验证。[具体核验](live_verification/0036_local_brain_joint/RESULT.json) · [部署接口](DEPLOYMENT.md)。

## 完整搜索与成本

共完成**7,896,659个有限全局参数元组**，不是独立方法数、独立统计假设数或唯一掩膜数；41个候选完整掩膜审计。包含镜像子集、温度/体积解码、预测一致性、STAPLE控制、非线性权重、MC-dropout、轴置换、近尺度/ROI、噪声平均、两方向512整数相位、官方RankSEG、局部MRF、原生3D DenseCRF、signed-distance形状融合、脑区约束及组件/孔洞组合，和28强度组及其有限融合。所有正负结果保留：[全分析表](ALL_COMPLETED_ANALYSES.tsv)、[已审计表](ALL_AUDITED_CANDIDATES.tsv)、各analyses目录及原矩阵SHA。

DenseCRF该有限网格最高是无边项控制，不能归功于边项；部分phase融合最高为alpha0控制，不能归功于相位。两512方向各138240次患者case-forward；1/64/128/256/512共享前缀，不能相加作独立NN预算。已完成生成器记录下界415,620次，另有fresh验证；失联旧节点/未提交失败计算不伪造精确总数。CPU后处理、I/O和统计成本另计，前向次数不是FLOPs。相位画布176×224×184，体素成本按实际填充形状单列，不能按native画布低估。

新worker1153287两A10080的20秒采样覆盖10.334小时，区间时间加权利用率GPU0 68.91%、GPU1 61.91%；不跨失联或采样外空档外推，不把某次100%称全程100%。[计算/采样口径](COMPUTE_SNAPSHOT.json)。合成NN-only硬件测量另见[批量吞吐与显存](HARDWARE_PROFILE.json)，不冒称真实病例端到端延迟，已核验生产批量仍B8/尾6。

## 边界、完整性和复现

唯一checkpoint SHA `02a60b2d37da0d8de8018c221e1c00566106bd21d0d713b95ac3d9f94d87fade`；原三载荷与index逐SHA一致。最终持久新增1.099GiB，实际计入主机保守余量的scratch 297.977GiB，满足2/300GiB上限及20/70GiB空闲门；原数据、模型、core8逐视图logits、全部分数/切点/审计掩膜保留。滚动清理只处理已归档无读者的本轮可重建派生文件，保留目录骨架；为组合28组输出，清理监督后来正常结束，没有模型重启。工程恢复5/6，旧失败与修复收据保留。[最终SHA与预算](FINAL_RUNTIME_AUDIT.json) · [权重/缓冲区不变及API输入契约](INFERENCE_CONTRACT_PROBE.json)。

联网原始论文/官方资料及借鉴范围见[WEB_REVIEW](WEB_REVIEW.md)，每轮网格和资源门见ROUND001–030。没有把其他架构、多权重或额外生成模型的收益归到本权重，也没有把当前有限网格的负结果扩称所有相关方法无效。全局参数直接复用270调优，病灶碎片过滤可能损害部分扫描，平均提升不表示每例都提高。

[机器结果](FINAL_RESULT.json) · [完成证据审计](COMPLETION_AUDIT.json)。科研仓库只同步源码和文本证据，不上传MRI/标签、权重、预测或数值数组；最终发布收据单独记录，历史缺失私有载荷的旧链接不冒称修复。
