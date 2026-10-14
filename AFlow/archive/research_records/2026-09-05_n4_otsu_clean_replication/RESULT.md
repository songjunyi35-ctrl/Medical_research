# N4仅用Otsu自动前景拟合：Stage A结果

状态：冻结开发门失败，当前实现关闭。

两个候选在任何E/F模型结果前预注册，共享同一canonical对照；E42 scans/17 subjects、F42/19，患者互斥。每fold教师与学生都只训练该fold930 fit，排除自身inner。不是独立外部或pristine blind证据。

| Fold | 推理 | Control Scan/Subject/Point | Candidate Scan/Subject/Point | Scan/Subject/Point差 |
|---|---|---|---|---|
| E | identity | 0.760810/0.766343/40/42 | 0.764925/0.769730/40/42 | +0.004115/+0.003388/+0 |
| E | flip8 | 0.766646/0.773611/40/42 | 0.772021/0.776761/40/42 | +0.005374/+0.003150/+0 |
| F | identity | 0.763052/0.746491/42/42 | 0.753817/0.739344/41/42 | -0.009235/-0.007148/-1 |
| F | flip8 | 0.769849/0.754045/42/42 | 0.766948/0.752529/42/42 | -0.002901/-0.001515/+0 |

## 区间与冻结门

10,000次患者簇bootstrap保留每位患者全部扫描；下面是逐比较95%区间，不作未注册的家族显著性声明。

- E identity：Scan CI [-0.004922423048089713, 0.01233259906599718]；Subject CI [-0.0064176458506308345, 0.014003179742118585]；失败门 ['scan_ci_safe', 'subject_ci_safe']。

- E flip8：Scan CI [-0.001193074105368741, 0.01115919975569884]；Subject CI [-0.0033031918621458127, 0.009311677350657977]；失败门 ['subject_ci_safe']。

- F identity：Scan CI [-0.01347002481244706, -0.0037018603199962877]；Subject CI [-0.013711007498330374, -0.0006535543205663661]；失败门 ['scan_ci_safe', 'scan_positive', 'subject_ci_safe', 'subject_positive']。

- F flip8：Scan CI [-0.007843847739865082, 0.0035283697261006717]；Subject CI [-0.008790058377847932, 0.006443879617674449]；失败门 ['scan_ci_safe', 'scan_positive', 'subject_ci_safe', 'subject_positive']。

## 因素与解释

唯一学生pair差异是N4拟合mask（正值→Otsu200），不删mask外组织。
工程/机制结果不能替代准确率；当前未通过全部预注册准确率门。两fold方向与区间按上表保留，不挑有利指标。

## 成本与下一决定

共享教师/对照14.426 GPU-hours；本候选新增6.783 GPU-hours；最高allocated显存20.167GiB。新缓存9.49GiB位于/tmp。部署4.70M参数，identity1×或固定Flip8 8×。
通过后仍须按时间窗条件完整Full972重训并另行冻结新外部确认；失败则关闭当前实现，不调参数、加seed、重测旧cohort。

本实验只新增E/F开发评测；无新fixed validation/outer/confirmation/test270/UCSD75/RHUH模型评测。原训练972包括此前已回收为train的旧selector/confirmation病例，不能把它们再称独立证据。

## 当前项目可比成绩与证据

本Stage A不改变旧排行榜。Reused test270总体（含同一checkpoint8镜像）最高点估计仍为Full972 KD168 Flip8：0.757270/0.741254/265；1×Scan最高GCSAM184为0.752003，1×Subject/Point最高Full972 KD168为0.737327/262。Mirror-uint8 UCSD75单列，不与内部test混排。

[七问](IDEA.md)、[协议](PREREG.md)、[原始结果](RAW_INNER.json)、[预访问收据](INNER_PREACCESS.json)、[机器结果与哈希](RESULT.json)。

[当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)。

- E fit930 candidate：[checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/n4_otsu_clean_20260905/formal/e_otsu/fixed.pth)，SHA256 `6601d19074a62cad53c568c20cacd0cbf3401b461b4b945b34634ce8e296d270`。不是Full972 final。

- F fit930 candidate：[checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/n4_otsu_clean_20260905/formal/f_otsu/fixed.pth)，SHA256 `51417050e83a57ae3faceba50cf13541dfb9e2b06c3fafa095eb537ee82c2b02`。不是Full972 final。
