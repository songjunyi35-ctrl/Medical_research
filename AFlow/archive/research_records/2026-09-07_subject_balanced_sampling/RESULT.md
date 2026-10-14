# 患者等概率采样：复用开发结果

状态：开发门失败，关闭当前策略。

本实验只将每个扫描的原生体积权重按患者归一化；其余网络、初值、主损失、辅助系数0、168轮、两seed和1×输出与预先固定native control一致。对照复用前一实验的固定结果，新候选从零训练。两seed共用同一78名复用开发患者，不是独立外部确认。

| seed | arm | Scan Dice | Subject Dice | Point /180 |
|---|---|---:|---:|---:|
| 2026090711 | candidate | 0.753475 | 0.740633 | 167 |
| 2026090711 | reused_control | 0.746762 | 0.733213 | 170 |
| 2026090712 | candidate | 0.755906 | 0.742725 | 168 |
| 2026090712 | reused_control | 0.744177 | 0.729318 | 163 |

患者聚类95%区间保留同患者全部扫描；Subject先患者内平均，Scan使用重采样患者贡献的总扫描均值。

- 2026090711：Scan差+0.006713，CI [0.0015553652737589326, 0.012000775264416582]；Subject差+0.007420，CI [0.0015311518691765562, 0.013552047004858318]；Point差-3。

- 2026090712：Scan差+0.011729，CI [0.005220724895760862, 0.019021115361279747]；Subject差+0.013406，CI [0.006627159794651781, 0.020810514007604877]；Point差+5。

新增两条训练合计4.028 GPU运行小时，峰值20.664 GiB；新增dev360 forwards，未重复control推理。缓存复用，部署仍4.70M/1×。

患者权重均匀是确定的采样性质，不能自动推出准确率提高。是否支持当前直觉由上述冻结开发门决定；失败不证明所有患者级采样无效。

close exact patient-equal policy; no temperature/mixture/epoch/seed search

## 人工解释：分割改善，但定位安全门未过

这不是“没有任何效果”的结果。两seed平均Scan差+0.009220523、Subject差+0.010413441；四个患者聚类配对Dice区间都高于0，两个seed的分割收益和区间安全门全部通过。唯一失败项是seed11 Point从170降到167（−3），超过预注册最多损失1例；seed12为163→168（+5）。不能把两seed的Point平均成+1后忽略单seed失败，也不能只挑seed12晋级。正式状态保持gate_failed，不放宽门或改采样混合比例。

因此直觉分两层：让每位患者获得相等机会确实增加了覆盖，且在本复用开发集支持更好的整块肿瘤重叠；但“整块涂色更准”不保证“分数最高的一个点一定落在肿瘤里”。这次没有证明自动定位同样稳定改善，更没有独立外部确认。只关闭当前满足全部目标的晋级路线，不把Dice正证据抹去，也不否定所有患者级采样。

完整168轮独立重建核查：两seed native/均衡策略的全部有效索引SHA与训练日志一致；对照每轮平均患者覆盖270.69/271.52，候选299.03/298.70；最小体积组抽样约44.77%/45.07%降至27.29%/27.38%。两候选与对应control初值、168轮增强随机序列和121步/轮均一致，辅助系数都0，最终导出error0。

揭示前固定体积分组的描述中，两seed四组Scan差均正；seed11定位净损失3例全部位于最小体积组，seed12该组却净增加3例，不宜下“所有小肿瘤都受损”的结论。两seed平均漏画比例分别减少0.004522/0.012157；多画/真值体积比变化方向不同，且可被极小真值放大，不能冒称假阳性率或单一机制因果。

给定保存的点坐标，720/720 Point布尔值和真值体素数与180份开发标签一致（包含重复control记录），命中数167/170、168/163不变；0新forward。这不独立验证未保存logit的argmax或预测mask重放。

新增训练14,502.28秒=4.028 GPU-hours，双卡训练阶段约2.02小时，包含合成检查/冻结/开发/归档的workflow约2.04小时。新增开发360 forwards耗时12.12秒；对照没有重跑。两份候选checkpoint及恢复文件保留，复用既有RC缓存，无新影像缓存。

补充证据：[完整抽样重建](../../reviews/2026-09-07_fit_subject_sampling/SAMPLER_REPLAY_FINAL.json)、[全部体积分组计数](../../reviews/2026-09-07_development_error_description/SUBJECT_SAMPLING.json)、[点标签核查](../../reviews/2026-09-07_development_error_description/SUBJECT_SAMPLING_POINT_AUDIT.json)。

未访问fixed validation、outer、confirmation、internal test270或外部新模型性能。旧内部总体仍Full KD168 Flip8（8×）0.757270/0.741254/265；1×最高Scan GCSAM0.752003，Subject/Point Full0.737327/262；这些历史test分数不与本dev180直接比较。Kaggle→官方Synapse parity限制继承。

[七问](IDEA.md)、[预注册](PREREG.md)、[设计冻结](DESIGN_FREEZE.json)、[机器结果](RESULT.json)、[raw](RAW_DEVELOPMENT.json)、[开发前收据](DEV_PREACCESS.json)、[验证](VERIFICATION_RECEIPT.json)。

Raw SHA `96785e393fd1033684014fda091f2de1545b42a0fce6fe226f79a8a10630017d`；RESULT JSON SHA `cfd44c8e727cae2250898e8b74084025fa370ab5b228c65cc161cfb00eb165e3`。

- [2026090711 checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907/seed2026090711_subject_balanced/fixed_epoch168.pth)：SHA `576cc8ff936695dfc109ec5c30af07fa57a539100368ecded080d3876b045efb`。

- [2026090712 checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907/seed2026090712_subject_balanced/fixed_epoch168.pth)：SHA `047268c8d60a2120ffbe734481c922b9c29387a3b0a024e8f8bc985a65434a11`。
