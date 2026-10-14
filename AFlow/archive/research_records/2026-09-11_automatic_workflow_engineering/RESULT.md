# 自动模型固定工作流：工程闭环结果

状态：工程执行完成；不是工作流搜索完整科研闭环，也不是精度晋级。

## 做了什么

只改变镜像集合，固定两个fit684/313、epoch168自动模型；训练侧前24图像，无标签。1/2/4/8视图均真实按需执行；16个在线/原子子集检查逐体素exact，历史identity/full8与原子融合也exact。没有训练、开发或test性能访问。

| Seed | 工作流 | 平均秒/例 | 配对中位加速（相对full8） |
|---|---|---:|---:|
| 2026090711 | full8 | 0.355547 | 1.000× |
| 2026090711 | identity | 0.045526 | 7.818× |
| 2026090711 | opposite2 | 0.089733 | 3.962× |
| 2026090711 | single_axis4 | 0.178367 | 1.993× |
| 2026090712 | full8 | 0.355883 | 1.000× |
| 2026090712 | identity | 0.045527 | 7.829× |
| 2026090712 | opposite2 | 0.089760 | 3.965× |
| 2026090712 | single_axis4 | 0.178468 | 1.994× |

工程速度门全部通过：True。GPU NVIDIA A100-SXM4-40GB；推理阶段墙钟74.52秒；共1654病例前向，其中1440计时前向、90预热、124一致性检查。

峰值allocated显存：0.738 GiB。0新训练、0新checkpoint、无持久大型预测缓存。

计时包含主机图像tensor到GPU、镜像前向融合、阈值/最高点、掩膜和点返回CPU；不包含文件读取、NIfTI预处理或模型加载。因此只能报告当前硬件的推理阶段收益，不叫端到端收益。重复测速不是独立患者，不给临床精度CI。

## 直觉支持与限制

实现确实只做选中的视图，数值结果与完整原子中的相同子集一致。速度门通过支持减少执行次数能节省本实现的推理耗时；没有标签，无法知道剪枝后Dice和Point是否安全，不能推荐部署或宣称超过排行榜。两个seed是同训练/开发来源，不是外部复制。

## 下一优先级

正式搜索前另冻协议：固定两个native模型和复用dev180/78，按患者分离搜索/选择/评估；明确重复开发而非新盲测，保留identity/full8参照；比较有限预算搜索与随机/贪心，并用完整有限空间作搜索质量参照。只在精度、点与实际成本联合门通过时才讨论后续确认，不自动开启test。旧subject-flip8负结果不复活，不用旧外部结果选视图。

本阶段完成的是该主线的执行和测速基础，不声称已经完成搜索算法比较、开发准确率验证或最终确认。

## 当前同队列纪录（仅历史背景）

复用非盲internal test270/119总体Scan0.757269662、Subject0.741253585、Point265/270（8×、一个checkpoint）；单次前向Scan0.752003176、Subject0.737266420、Point252/270。不是官方隐藏榜；本工程未改变排名。总体Scan绝对95%CI[0.739089,0.774152]，Subject[0.714109,0.766152]；不能将历史CI套在本轮测速上。

## 证据

- [七问/冻结边界](IDEA.md)、[前访问收据](PREACCESS.json)、[原始测速](RAW_BENCHMARK.json)、[机器结果](RESULT.json)、[验证](VERIFICATION.json)
- [当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)
- [官方镜像推理实现](https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/inference/predict_from_raw_data.py)仅支持镜像融合实现依据，不支持精度剪枝收益。
- [固定seed2026090711 checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907/seed2026090711_control/fixed_epoch168.pth)，SHA `fdad97d9abc736638bcd9911fc82ddca1a434fd0c9a46c3e9306d608276b3d6e`
- [固定seed2026090712 checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907/seed2026090712_control/fixed_epoch168.pth)，SHA `5b014fb256c272b828e01322b7985f10d8ac5c53f19fa475761c25537f697e43`
