# 保护峰值的小孤岛清理：开发结果

状态：completed_development_gate_failed。

只增加固定26邻接小组件清理（max(8,最大块1%)，并保护最大块/最高logit点），其余checkpoint/阈值/输入/模型不变。两个旧native模型都只由fit684训练；同一dev180/78已复用，非盲、非外部、不是两份独立cohort。无新训练。

| seed | 输出 | Scan Dice | Subject Dice | Point /180 |
|---|---|---:|---:|---:|
| 2026090711 | base | 0.746762 | 0.733213 | 170 |
| 2026090711 | clean | 0.746669 | 0.733122 | 170 |
| 2026090712 | base | 0.744177 | 0.729318 | 163 |
| 2026090712 | clean | 0.744057 | 0.729148 | 163 |

2026090711：Scan差-0.000093，患者簇95%CI [-0.0002705513775007457, 6.780922801216407e-05]；Subject差-0.000091，CI [-0.0002483689943531265, 5.371384060531597e-05]；Point差+0，180点位置全相同。改善/恶化/相同=46/46/88。
清理涉及94例，删真阳性3805体素、假阳性4401体素；仅占该模型原假阳性体素的0.345%。

2026090712：Scan差-0.000121，患者簇95%CI [-0.0003187134382961463, 6.214996833725933e-05]；Subject差-0.000170，CI [-0.00040086726981864595, 3.720381365948643e-05]；Point差+0，180点位置全相同。改善/恶化/相同=59/46/75。
清理涉及106例，删真阳性5071体素、假阳性4282体素；仅占该模型原假阳性体素的0.376%。

区间为固定seed的10,000次患者簇百分位95%CI，未校正多比较；Point不变是算子设计及逐例验证，不代表Dice必升。任何失败门不按比例/连接性/阈值救回。

成本：新训练0，开发360forwards/31.63s，额外峰值4.815GiB；部署模型仍1×，但多CPU组件计数和传输延迟，无新影像cache/新checkpoint。

close_fixed_cleanup_rule_no_size_fraction_connectivity_search

未访问弱形变新checkpoint、test270、fixed validation、outer、confirmation或external。弱形变训练/门保持原样，不与本规则捆绑。旧榜overall仍0.757269662（Full972KD168八镜像8×）、1×Scan0.752003176（GCSAM184），同test270/119，不能用开发数值直接比较。

[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[preaccess](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[验证](VERIFICATION_RECEIPT.json)。

Raw SHA `2f85c8e96be92c51475ad6cf7ae7612b349b305698769b0d887e52ea1a8ddad4`；RESULT JSON SHA `a8d60af92be3bc71e4f591eee90144d858b44bd1601b14be4cfa4130795ac2f0`。

- [2026090711原模型checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907/seed2026090711_control/fixed_epoch168.pth)；清理需额外固定规则，不存在新训练权重。

- [2026090712原模型checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907/seed2026090712_control/fixed_epoch168.pth)；清理需额外固定规则，不存在新训练权重。
