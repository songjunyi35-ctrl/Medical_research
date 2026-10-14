# 训练期切除腔辅助监督：复用开发结果

状态：开发门失败，关闭当前配方。

本实验只测试额外训练监督是否改善WT；两臂均从零、无教师、同结构、同fit和计算预算，唯一因素为辅助权重0→0.1。开发集180 scans/78人来自用户批准的复用开发池，两个seed共享同一批患者，不能称两次独立泛化验证。

| seed/arm | Scan Dice | Subject Dice | Point /180 |
|---|---:|---:|---:|
| seed2026090711_candidate | 0.744044 | 0.729224 | 166 |
| seed2026090711_control | 0.746762 | 0.733213 | 170 |
| seed2026090712_candidate | 0.739218 | 0.724989 | 161 |
| seed2026090712_control | 0.744177 | 0.729318 | 163 |

配对95%区间按患者重采样，每位患者全部扫描一起进入Scan统计；Subject先患者内平均。

- Seed 2026090711：Scan差-0.002718，cluster CI [-0.010668538591962744, 0.003736539000601968]；Subject差-0.003989，CI [-0.012727358520559185, 0.003427021006036891]；Point差-4。

- Seed 2026090712：Scan差-0.004959，cluster CI [-0.009987977999539639, 0.00014995837383397767]；Subject差-0.004329，CI [-0.00987965915596551, 0.0014073457436917037]；Point差-2。

## 成本、解释和下一步

四条训练合计8.064 GPU运行小时，峰值20.664 GiB。辅助uint8缓存约1.34GiB；部署裁除辅助输出，4.70M参数、1×前向。开发评测720 forwards。

结果只能说明当前0.1辅助监督在该复用开发协议的表现。辅助任务学得更好不自动意味着WT更好；不能由此更改已揭示外部队列后处理或重跑test。

close weight0.1/current recipe without epoch/seed/weight search

### 人工解释与跨seed复核

两seed平均Scan差−0.003838836、Subject差−0.004159115；前者已另从两条原始比较重算并与raw字段核对。每seed全部五个性能门都失败，Point分别少4/2例；四个Dice配对区间均跨0，结论是未达到预先要求的提升和安全性，而非统计上确定有害。

直觉仅有一半成立：两候选确实学会辅助任务，末10轮aux loss为0.15495/0.15064，对照1.00095/0.99719；但主任务开发没有受益。第一seed末10轮训练Dice0.78557/0.78511接近，第二seed候选0.77772低于对照0.78485。四条训练导出误差均0、配对工程门通过；结果不能归咎于未接上辅助监督，也不能推广为“所有背景细分方法都无效”。

四训练合计29,031.54秒；按每pair较慢臂计算约4.05小时，包含两批启动/核验的实际总墙钟约4.07小时。唯一开发评测720 forwards耗时23.12秒。辅助缓存与四份最终checkpoint均保留，无删除。下一患者等概率采样方案已在本结果揭示前固定，复用两control；本负结果不改变其内容。

历史内部总体点估计仍为Full KD168固定Flip8（8×）Scan0.757270/Subject0.741254/Point265；内部1×最高Scan为GCSAM0.752003、最高Subject/Point为Full0.737327/262。这些旧test结果与本开发集不直接比较。

本实验没有访问fixed validation108、outer、confirmation、internal test270或任何外部模型性能。训练源Kaggle→官方Synapse字节parity限制继承。

## 证据

[七问](IDEA.md)、[预注册](PREREG.md)、[机器结果](RESULT.json)、[raw](RAW_DEVELOPMENT.json)、[开发前收据](DEV_PREACCESS.json)、[独立重算](VERIFICATION_RECEIPT.json)。

Raw SHA256 `80e9bd449d7c72a61de0ecdbdfbc5bb3a61814e7770bbcc1c983ac097b4cd2c6`；RESULT JSON SHA256 `a59c919f358890cfe1f7552796d8259b0088cff99522494489dadfa1ecf62c1e`。

- [seed2026090711_candidate checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907/seed2026090711_candidate/fixed_epoch168.pth)：SHA256 `b493039d63428368d17b1fcc33d67231c5d91667f5b6acb02274ac054a8750f3`。

- [seed2026090711_control checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907/seed2026090711_control/fixed_epoch168.pth)：SHA256 `fdad97d9abc736638bcd9911fc82ddca1a434fd0c9a46c3e9306d608276b3d6e`。

- [seed2026090712_candidate checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907/seed2026090712_candidate/fixed_epoch168.pth)：SHA256 `475b0bb269d98c2532374ea554ec1150e1f4326556100fc270e487bc577600e7`。

- [seed2026090712_control checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907/seed2026090712_control/fixed_epoch168.pth)：SHA256 `5b014fb256c272b828e01322b7985f10d8ac5c53f19fa475761c25537f697e43`。
