# 患者模型固定八镜像：开发结果

状态：completed_development_gate_failed。

同两个固定患者采样checkpoint，只把1次原输出改成同一模型8镜像还原raw-logit等权平均；128³/阈值0/argmax/原hard GT不变，fit684/313、dev180/78，无训练或多模型融合。历史1×采样配方的失败结论保留，本新应用单独验证。

| seed | 输出 | Scan Dice | Subject Dice | Point /180 |
|---|---|---:|---:|---:|
| 2026090711 | flip8 | 0.759244 | 0.745888 | 169 |
| 2026090711 | identity | 0.753475 | 0.740633 | 167 |
| 2026090712 | flip8 | 0.758990 | 0.745968 | 160 |
| 2026090712 | identity | 0.755906 | 0.742725 | 168 |

- 2026090711 Flip8−identity：Scan +0.005769 CI [0.003818253381104681, 0.007667861598497258]；Subject +0.005255 CI [0.0026660287379831078, 0.007639378366016113]；Point +2。

- 2026090712 Flip8−identity：Scan +0.003084 CI [0.0005812212260905095, 0.005469962937227189]；Subject +0.003244 CI [0.0002310871018209463, 0.006332766345659574]；Point -8。

10k患者簇百分位95%CI，两个seed共用78人，未多比较校正；本提案基于已揭示开发和历史推理证据，不是新盲确认。

成本：0训练、2880forwards/54.41s、peak5.175GiB，部署同4.70M/8×而非1×。

close_fixed_subject_flip8_no_view_weight_threshold_seed_search

当前没有test/oldconfirmation/fixed validation/outer/external新访问；不能把dev分数与旧test270榜0.7572696616942132直接比较。

[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[开发前收据](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[验证](VERIFICATION_RECEIPT.json)。

Raw SHA `124e6c3634db004992d558f627a8a9097a6897cb7aaadafee06a173cf3e83afb`；RESULT SHA `842cb4d7573204634516511f73d5810eb6afe7df8ea776f62c19fdd555926cb9`。
