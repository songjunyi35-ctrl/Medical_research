# 图像模糊患者抽样层：复用开发结果

状态：completed_reused_development_gate_failed；唯一入选层：无。

每个固定抽样层仅追加p0.2/sigma0.5—1输入体素的仅图像模糊，原模型、初值、168轮、基础增强、主损失、1×输出不变。两个seed共用同一180扫描/78患者reused development，不是两份独立确认。无教师；历史四个无追加旋转对照复用已揭示记录。

| seed/新增臂 | Scan Dice | Subject Dice | Point /180 |
|---|---:|---:|---:|
| 2026090711_subject_blur | 0.747191 | 0.734376 | 170 |
| 2026090712_subject_blur | 0.752821 | 0.741177 | 165 |

原native control11/12：Scan0.746762/0.744177、Subject0.733213/0.729318、Point170/163；原subject control11/12：Scan0.753475/0.755906、Subject0.740633/0.742725、Point167/168。

两Dice采用10,000次患者簇配对百分位95%CI；探索性两候选筛选，未校正多比较。Point列为计数差，不据平均值掩盖单seed失败。

- 2026090711_subject_blur versus_native：Scan +0.000429 CI [-0.004729623614770604, 0.005866604427713323]；Subject +0.001163 CI [-0.0046028104871956525, 0.007010879917160672]；Point +0。

- 2026090711_subject_blur versus_own_sampling_control：Scan -0.006284 CI [-0.011025260977284863, -0.0016046553176505907]；Subject -0.006258 CI [-0.010895212804331166, -0.0017547816636191928]；Point +3。

- 2026090712_subject_blur versus_native：Scan +0.008643 CI [0.0029925556312216405, 0.014585936641484405]；Subject +0.011859 CI [0.0063398337529694095, 0.017922450297547417]；Point +2。

- 2026090712_subject_blur versus_own_sampling_control：Scan -0.003085 CI [-0.008922963339260665, 0.0025105293123061107]；Subject -0.001548 CI [-0.008344698339039938, 0.005412431515993524]；Point -3。

新增两训练4.330 GPU小时，峰值20.664GiB；开发360 forwards/11.33s；无新影像cache，模型仍4.70M/1×。

工程证明实现的形变/随机序列如预期，不自动证明泛化获益。冻结联合门决定本配方是否支持晋级；失败不证明所有空间增强无效。详细机制解释与后续决定在INTERPRETATION.md完成后记录。

没有新增test/outer/confirmation/fixed validation/external性能；旧test270/119 overall仍Full972KD168 Flip8 Scan0.757269662，1×Scan仍GCSAM0.752003176。它们不与本dev180分数直接比较。训练源Kaggle→官方字节parity未知限制保留。

[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[preaccess](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[验证](VERIFICATION_RECEIPT.json)、[全训练重放](TRAINING_REPLAY_FINAL.json)。

Raw SHA `60e013895ce04f954514b4aa87fc60a8baca479a3b836da9e4b3aeadbba7a00d`；RESULT JSON SHA `64ef3c85ed4a648b684621d7f14d1cdab0365ff9c2d92690e060995c3991a942`。

- [2026090711_subject_blur checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/image_blur_20260910/seed2026090711_subject_blur/fixed_epoch168.pth)：SHA `bc9fb2cc7cd574032e7091316ceaa06f1e665a014ab4b30296fbe46733593ba6`。

- [2026090712_subject_blur checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/image_blur_20260910/seed2026090712_subject_blur/fixed_epoch168.pth)：SHA `f611c3ba152ae9086bf813266d90167830c7b34cffc3f2a548a96e7dad8915a9`。
