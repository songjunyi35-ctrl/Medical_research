# 弱弹性形变双抽样层：复用开发结果

状态：completed_reused_development_gate_failed；唯一入选层：无。

每个固定抽样层仅追加p0.2/最大2输入体素的平滑形变，原模型、初值、168轮、基础增强、主损失、1×输出不变。两个seed共用同一180扫描/78患者reused development，不是两份独立确认。无教师；历史四个无形变对照复用已揭示记录。

| seed/新增臂 | Scan Dice | Subject Dice | Point /180 |
|---|---:|---:|---:|
| 2026090711_native_elastic | 0.745530 | 0.731086 | 172 |
| 2026090711_subject_elastic | 0.742229 | 0.732752 | 172 |
| 2026090712_native_elastic | 0.744132 | 0.730980 | 167 |
| 2026090712_subject_elastic | 0.751126 | 0.739671 | 165 |

原native control11/12：Scan0.746762/0.744177、Subject0.733213/0.729318、Point170/163；原subject control11/12：Scan0.753475/0.755906、Subject0.740633/0.742725、Point167/168。

两Dice采用10,000次患者簇配对百分位95%CI；探索性两候选筛选，未校正多比较。Point列为计数差，不据平均值掩盖单seed失败。

- 2026090711_native_elastic versus_native：Scan -0.001232 CI [-0.00668011230049938, 0.0043232206555765896]；Subject -0.002127 CI [-0.008205531742031336, 0.0041469071155517175]；Point +2。

- 2026090711_native_elastic versus_own_sampling_control：Scan -0.001232 CI [-0.00668011230049938, 0.0043232206555765896]；Subject -0.002127 CI [-0.008205531742031336, 0.0041469071155517175]；Point +2。

- 2026090711_subject_elastic versus_native：Scan -0.004533 CI [-0.011417640168590783, 0.0026002048548480917]；Subject -0.000461 CI [-0.007075657545962269, 0.006302345193568108]；Point +2。

- 2026090711_subject_elastic versus_own_sampling_control：Scan -0.011245 CI [-0.017368119095378182, -0.00536187625584284]；Subject -0.007882 CI [-0.012822737456769775, -0.0031188451468500194]；Point +5。

- 2026090712_native_elastic versus_native：Scan -0.000046 CI [-0.005324285889845311, 0.004786755838777853]；Subject +0.001661 CI [-0.0027389637751959557, 0.006035124372625505]；Point +4。

- 2026090712_native_elastic versus_own_sampling_control：Scan -0.000046 CI [-0.005324285889845311, 0.004786755838777853]；Subject +0.001661 CI [-0.0027389637751959557, 0.006035124372625505]；Point +4。

- 2026090712_subject_elastic versus_native：Scan +0.006948 CI [0.0010873631131578824, 0.012958366993285503]；Subject +0.010353 CI [0.004550314056493696, 0.016323164117862995]；Point +2。

- 2026090712_subject_elastic versus_own_sampling_control：Scan -0.004780 CI [-0.009261050504705173, -0.0003870503407520882]；Subject -0.003054 CI [-0.008137320340234016, 0.002333826199550761]；Point -3。

新增四训练8.627 GPU小时，峰值20.664GiB；开发720 forwards/21.84s；无新影像cache，模型仍4.70M/1×。

工程证明实现的形变/随机序列如预期，不自动证明泛化获益。冻结联合门决定本配方是否支持晋级；失败不证明所有空间增强无效。详细机制解释与后续决定在INTERPRETATION.md完成后记录。

没有新增test/outer/confirmation/fixed validation/external性能；旧test270/119 overall仍Full972KD168 Flip8 Scan0.757269662，1×Scan仍GCSAM0.752003176。它们不与本dev180分数直接比较。训练源Kaggle→官方字节parity未知限制保留。

[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[preaccess](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[验证](VERIFICATION_RECEIPT.json)、[全训练重放](TRAINING_REPLAY_FINAL.json)。

Raw SHA `c9ff58c495fb96f9fa607820140351c25bfe81e02e0290c6f2e4084145ae399e`；RESULT JSON SHA `b78a49c568b61a047f5361b0a5f147ff45a6fdbed77aca99403669e1f5a4ba76`。

- [2026090711_native_elastic checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/weak_elastic_20260910/seed2026090711_native_elastic/fixed_epoch168.pth)：SHA `12bd86129ac5d387c4ffd608012f7aabdccbb30e2b5780b90c24e8f1455683d7`。

- [2026090711_subject_elastic checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/weak_elastic_20260910/seed2026090711_subject_elastic/fixed_epoch168.pth)：SHA `273537ab9cc0de7e2bcd97ce1885ec9319436275f4f30ebc897c17ac41775e65`。

- [2026090712_native_elastic checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/weak_elastic_20260910/seed2026090712_native_elastic/fixed_epoch168.pth)：SHA `e8e75534e582ba692624ae403e13c9d5264dd42b462d6691ca31fd03f4aad93c`。

- [2026090712_subject_elastic checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/weak_elastic_20260910/seed2026090712_subject_elastic/fixed_epoch168.pth)：SHA `4fc95ae79768192c5033804b214f70f55ee36659492fe3934e62bc8141f795fe`。
