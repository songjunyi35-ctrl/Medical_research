# 取消额外N4校正：Stage A结果

状态：冻结开发门失败，当前实现关闭。

两个候选在任何E/F模型结果前预注册，共享同一canonical对照；E42 scans/17 subjects、F42/19，患者互斥。每fold教师与学生都只训练该fold930 fit，排除自身inner。不是独立外部或pristine blind证据。

| Fold | 推理 | Control Scan/Subject/Point | Candidate Scan/Subject/Point | Scan/Subject/Point差 |
|---|---|---|---|---|
| E | identity | 0.760810/0.766343/40/42 | 0.778854/0.779039/40/42 | +0.018044/+0.012696/+0 |
| E | flip8 | 0.766646/0.773611/40/42 | 0.784280/0.785318/41/42 | +0.017634/+0.011707/+1 |
| F | identity | 0.763052/0.746491/42/42 | 0.763649/0.748730/40/42 | +0.000597/+0.002239/-2 |
| F | flip8 | 0.769849/0.754045/42/42 | 0.770036/0.755637/42/42 | +0.000187/+0.001592/+0 |

## 区间与冻结门

10,000次患者簇bootstrap保留每位患者全部扫描；下面是逐比较95%区间，不作未注册的家族显著性声明。

- E identity：Scan CI [0.005431424321486961, 0.029630890696942323]；Subject CI [0.00353870457673654, 0.02196303869255041]；失败门 []。

- E flip8：Scan CI [0.004484772418762747, 0.028602978201245143]；Subject CI [-0.0003441510651530754, 0.02335606112794552]；失败门 []。

- F identity：Scan CI [-0.0068399565346707715, 0.008472690209669571]；Subject CI [-0.00792088846985928, 0.01407946156394375]；失败门 ['point_within_one', 'scan_ci_safe', 'subject_ci_safe']。

- F flip8：Scan CI [-0.007964852101813111, 0.008943651043488175]；Subject CI [-0.009649284338798112, 0.014830018005016054]；失败门 ['scan_ci_safe', 'subject_ci_safe']。

## 因素与解释

唯一学生pair差异是是否执行N4；原正值支持、标签/几何/模型/教师权重/seed/损失/采样/168轮/阈值固定。
工程/机制结果不能替代准确率；当前未通过全部预注册准确率门。两fold方向与区间按上表保留，不挑有利指标。

## 成本与下一决定

共享教师/对照14.426 GPU-hours；本候选新增6.784 GPU-hours；最高allocated显存20.167GiB。新缓存9.49GiB位于/tmp。部署4.70M参数，identity1×或固定Flip8 8×。
通过后仍须按时间窗条件完整Full972重训并另行冻结新外部确认；失败则关闭当前实现，不调参数、加seed、重测旧cohort。

本实验只新增E/F开发评测；无新fixed validation/outer/confirmation/test270/UCSD75/RHUH模型评测。原训练972包括此前已回收为train的旧selector/confirmation病例，不能把它们再称独立证据。

## 当前项目可比成绩与证据

本Stage A不改变旧排行榜。Reused test270总体（含同一checkpoint8镜像）最高点估计仍为Full972 KD168 Flip8：0.757270/0.741254/265；1×Scan最高GCSAM184为0.752003，1×Subject/Point最高Full972 KD168为0.737327/262。Mirror-uint8 UCSD75单列，不与内部test混排。

[七问](IDEA.md)、[协议](PREREG.md)、[原始结果](RAW_INNER.json)、[预访问收据](INNER_PREACCESS.json)、[机器结果与哈希](RESULT.json)。

[当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)。

- E fit930 candidate：[checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/no_n4_clean_20260905/formal/e_no_n4/fixed.pth)，SHA256 `e61accad8afac1209545205c06359eb342936ab09b796576aa245b27859d5437`。不是Full972 final。

- F fit930 candidate：[checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/no_n4_clean_20260905/formal/f_no_n4/fixed.pth)，SHA256 `0c3753585140367fecb4684e529b2eeca3438271d21175d1377111b780c12912`。不是Full972 final。
