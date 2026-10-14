# 不同数据配方等权融合：开发结果

状态：completed_development_gate_failed。

仅将同seed两个固定168模型的raw logits等权平均；阈值0、平均分数最高点、原hard GT不变。源模型分别由患者均衡和体积占比标签训练，均仅fit684/313、无教师；两个原单模型失败结论保留。此提案产生于已揭示dev之后，复用180/78、非独立确认。

| seed | 输出 | Scan Dice | Subject Dice | Point /180 |
|---|---|---:|---:|---:|
| 2026090711 | ensemble | 0.754116 | 0.740313 | 174 |
| 2026090711 | occupancy | 0.746659 | 0.732690 | 175 |
| 2026090711 | sampling | 0.753475 | 0.740633 | 167 |
| 2026090712 | ensemble | 0.750449 | 0.736772 | 167 |
| 2026090712 | occupancy | 0.739852 | 0.726270 | 167 |
| 2026090712 | sampling | 0.755906 | 0.742725 | 168 |

- 2026090711 ensemble−native：Scan +0.007354 CI [0.0034292584265822484, 0.011192236314894738]；Subject +0.007101 CI [0.002580883817215205, 0.01158600025505401]；Point +4。

- 2026090711 ensemble−occupancy：Scan +0.007458 CI [0.0034014495567778334, 0.011347240847385408]；Subject +0.007623 CI [0.0034909354553479366, 0.011389642336141776]；Point -1。

- 2026090711 ensemble−sampling：Scan +0.000642 CI [-0.00215777526598611, 0.003273157167609235]；Subject -0.000320 CI [-0.0033361621462096487, 0.0024194462975393434]；Point +7。

- 2026090712 ensemble−native：Scan +0.006272 CI [0.0014512802247943605, 0.010960445593684545]；Subject +0.007454 CI [0.002921377774678187, 0.01185388686966253]；Point +4。

- 2026090712 ensemble−occupancy：Scan +0.010597 CI [0.008234999073173632, 0.012914557638256451]；Subject +0.010502 CI [0.007640675419788642, 0.013233254055821321]；Point +0。

- 2026090712 ensemble−sampling：Scan -0.005457 CI [-0.009935554535112984, -0.0012385300373015252]；Subject -0.005952 CI [-0.010620699124479476, -0.0016068366018691775]；Point -1。

10k患者簇百分位95%CI，未多比较校正；不能用一个模型Dice和另一个Point拼成绩。所有成员原输出逐例parity通过，融合是否兼得由冻结联合门判断。

新训练0；dev720成员forwards/17.06s、峰值4.849GiB。部署两模型/2×（如固定八镜像为16×），不是1×提分；没有新checkpoint。

close_fixed_equal_fusion_no_weight_member_seed_search

没有新增test/oldconfirmation/fixed validation/outer/external。旧test270/119整体0.7572696616942132、1×0.7520031755617648不与此开发分数混比。

[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[开发前冻结](DEV_PREACCESS.json)、[机器结果](RESULT.json)、[独立验证](VERIFICATION_RECEIPT.json)。

Raw SHA `4840f18597b651207484c18a309b62e074a153c74b24dc21cab98e304b2b5552`；RESULT SHA `066aec3933725d083452efa1552a32e2f276d717bb7a4228a78669dba1f30e7a`。

- [2026090711_occupancy checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/fractional_occupancy_20260910/seed2026090711_fractional_occupancy/fixed_epoch168.pth) SHA `ea144d90429c733e998d4f38c980e8cbd542aa09e2ec1febeec8030941449d1e`。

- [2026090711_sampling checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907/seed2026090711_subject_balanced/fixed_epoch168.pth) SHA `576cc8ff936695dfc109ec5c30af07fa57a539100368ecded080d3876b045efb`。

- [2026090712_occupancy checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/fractional_occupancy_20260910/seed2026090712_fractional_occupancy/fixed_epoch168.pth) SHA `1ed5c7a850997076f4222c69e38d5c8cf676abd0d5631731dc4ea9e33ef85dae`。

- [2026090712_sampling checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907/seed2026090712_subject_balanced/fixed_epoch168.pth) SHA `047268c8d60a2120ffbe734481c922b9c29387a3b0a024e8f8bc985a65434a11`。
