# 小角度输入网格旋转双抽样层：复用开发结果

状态：completed_reused_development_gate_failed；唯一入选层：无。

每个固定抽样层仅追加p0.2/每轴±15度的整体旋转，原模型、初值、168轮、基础增强、主损失、1×输出不变。两个seed共用同一180扫描/78患者reused development，不是两份独立确认。无教师；历史四个无追加旋转对照复用已揭示记录。

| seed/新增臂 | Scan Dice | Subject Dice | Point /180 |
|---|---:|---:|---:|
| 2026090711_native_rotation | 0.751434 | 0.736974 | 167 |
| 2026090711_subject_rotation | 0.747543 | 0.733799 | 171 |
| 2026090712_native_rotation | 0.742950 | 0.728538 | 169 |
| 2026090712_subject_rotation | 0.749583 | 0.736780 | 159 |

原native control11/12：Scan0.746762/0.744177、Subject0.733213/0.729318、Point170/163；原subject control11/12：Scan0.753475/0.755906、Subject0.740633/0.742725、Point167/168。

两Dice采用10,000次患者簇配对百分位95%CI；探索性两候选筛选，未校正多比较。Point列为计数差，不据平均值掩盖单seed失败。

- 2026090711_native_rotation versus_native：Scan +0.004672 CI [-0.0001769323693566359, 0.009404105725512311]；Subject +0.003761 CI [-0.0007824524387966389, 0.0080267282732672]；Point -3。

- 2026090711_native_rotation versus_own_sampling_control：Scan +0.004672 CI [-0.0001769323693566359, 0.009404105725512311]；Subject +0.003761 CI [-0.0007824524387966389, 0.0080267282732672]；Point -3。

- 2026090711_subject_rotation versus_native：Scan +0.000780 CI [-0.004905567777308139, 0.006185964623918987]；Subject +0.000586 CI [-0.005710500105031027, 0.006485007040210358]；Point +1。

- 2026090711_subject_rotation versus_own_sampling_control：Scan -0.005932 CI [-0.011325460142671068, -0.0007647756578569129]；Subject -0.006835 CI [-0.01305546347802885, -0.0008415691422485338]；Point +4。

- 2026090712_native_rotation versus_native：Scan -0.001228 CI [-0.005883751850615039, 0.0033131231811615426]；Subject -0.000780 CI [-0.0052133506180896474, 0.003732050830042173]；Point +6。

- 2026090712_native_rotation versus_own_sampling_control：Scan -0.001228 CI [-0.005883751850615039, 0.0033131231811615426]；Subject -0.000780 CI [-0.0052133506180896474, 0.003732050830042173]；Point +6。

- 2026090712_subject_rotation versus_native：Scan +0.005405 CI [-0.0007464398758984856, 0.011679657183510541]；Subject +0.007462 CI [0.001574056490755848, 0.01355174309915373]；Point -4。

- 2026090712_subject_rotation versus_own_sampling_control：Scan -0.006323 CI [-0.011061129782726557, -0.001908067419385177]；Subject -0.005945 CI [-0.010815354953533677, -0.0015183947933596759]；Point -9。

新增四训练8.646 GPU小时，峰值20.695GiB；开发720 forwards/22.62s；无新影像cache，模型仍4.70M/1×。

工程证明实现的形变/随机序列如预期，不自动证明泛化获益。冻结联合门决定本配方是否支持晋级；失败不证明所有空间增强无效。详细机制解释与后续决定在INTERPRETATION.md完成后记录。

没有新增test/outer/confirmation/fixed validation/external性能；旧test270/119 overall仍Full972KD168 Flip8 Scan0.757269662，1×Scan仍GCSAM0.752003176。它们不与本dev180分数直接比较。训练源Kaggle→官方字节parity未知限制保留。

[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[preaccess](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[验证](VERIFICATION_RECEIPT.json)、[全训练重放](TRAINING_REPLAY_FINAL.json)。

Raw SHA `696137e01047cf3261c11827589a4aabdbc72cce282d07a97243801c0c63e9de`；RESULT JSON SHA `03dfc8872c889631321e39c325a9f399562ba85ed704a634e13c5a3d08491bdd`。

- [2026090711_native_rotation checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/small_rotation_20260910/seed2026090711_native_rotation/fixed_epoch168.pth)：SHA `3dd1460fec01dc6c919f9666c0e8f8546bf645050eab7c90f8186c3defcafab9`。

- [2026090711_subject_rotation checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/small_rotation_20260910/seed2026090711_subject_rotation/fixed_epoch168.pth)：SHA `760b165160ad1bbe586024ae246916a4560d51652efd4835e3c114cf0242166a`。

- [2026090712_native_rotation checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/small_rotation_20260910/seed2026090712_native_rotation/fixed_epoch168.pth)：SHA `00fed7d67f5353d2fa8d7a81ecd2be06ea277ca6bb3949bc1954bda3d97898c3`。

- [2026090712_subject_rotation checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/small_rotation_20260910/seed2026090712_subject_rotation/fixed_epoch168.pth)：SHA `1c7d451141e889f8ff0684a57ac12b4a5ef391482cbbe2933dae9002eaf3de45`。
