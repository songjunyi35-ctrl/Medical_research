# 精确体积占比训练标签：原硬GT开发结果

状态：completed_reused_development_gate_failed。

唯一改变为训练WT target的体积占比表示，float64计算/float16缓存/float32训练；图像、native抽样、初始化、原增强、168轮、主损失、阈值0和1×输出不变，无教师。开发仍原硬128³ GT，不按新target评分。两seed共用同一180扫描/78患者reused development，非外部或独立确认。

| seed | arm | Scan Dice | Subject Dice | Point /180 |
|---|---|---:|---:|---:|
| 2026090711 | candidate | 0.746659 | 0.732690 | 175 |
| 2026090711 | reused_control | 0.746762 | 0.733213 | 170 |
| 2026090712 | candidate | 0.739852 | 0.726270 | 167 |
| 2026090712 | reused_control | 0.744177 | 0.729318 | 163 |

2026090711：Scan差-0.000104，簇95%CI [-0.004424928356707556, 0.004283946076641247]；Subject差-0.000523，CI [-0.0048222630116032264, 0.003983168821690982]；Point差+5。

2026090712：Scan差-0.004325，簇95%CI [-0.00900678634259089, -2.593773819401302e-05]；Subject差-0.003048，CI [-0.006964198724375176, 0.0007130225949826978]；Point差+4。

10,000次患者簇百分位95%CI，两个固定seed、同一78人，未多比较校正；不取较好seed或训练epoch。质量守恒是工程性质，不自动证明分割提升或概率校准。

新增两训练4.263GPU小时、峰值20.664GiB；开发360forwards/10.99s。临时soft target cache2.672GiB，原影像不复制；部署4.70M/1×。

close_exact_occupancy_target_original_loss_pairing_no_kernel_threshold_seed_search

目前无新增test/outer/confirmation/fixed validation/external性能；旧test270/119 overall Full972KD168 Flip8 Scan0.7572696616942132，1×Scan GCSAM0.7520031755617648，不能和此dev180混比。来源镜像/官方字节parity未证明限制仍保留。

[七问](IDEA.md)、[预注册](PREREG.md)、[cache](CACHE_RECEIPT.json)、[raw](RAW_DEVELOPMENT.json)、[preaccess](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[验证](VERIFICATION_RECEIPT.json)。

Raw SHA `b1474c2a6e0356559f6e5e15639fa9d705d5cce08c8c2a3556e142b0a06a1647`；RESULT JSON SHA `91c6f2e9b529d25f46f4b2faace340a2adf42a84ee55c6a400d0621053107902`。

- [2026090711 checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/fractional_occupancy_20260910/seed2026090711_fractional_occupancy/fixed_epoch168.pth) SHA `ea144d90429c733e998d4f38c980e8cbd542aa09e2ec1febeec8030941449d1e`。

- [2026090712 checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/fractional_occupancy_20260910/seed2026090712_fractional_occupancy/fixed_epoch168.pth) SHA `1ed5c7a850997076f4222c69e38d5c8cf676abd0d5631731dc4ea9e33ef85dae`。
