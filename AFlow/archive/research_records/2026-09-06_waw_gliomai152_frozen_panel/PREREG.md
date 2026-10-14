# WAW-GliomAI152冻结checkpoint面板：预注册

状态：冻结于大MRI包访问、cache、checkpoint加载和性能访问之前。

## 冻结输入与身份

- Zenodo final record：`22205631`，publication `2026-09-01`，open access，CC BY 4.0。
- 来源审计：`archive/reviews/2026-09-06_waw_gliomai_source_audit/SOURCE_AUDIT.json` SHA-256 `62503b3e680f0fc6922f20d6732a608d226429357bb8e5bb52da67e7ade7d3e7`。
- cohort：final-release preop coreg T1c metadata与preop mask严格交集，152 scans=152 subjects；`FROZEN_COHORT.json` SHA-256 `25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a`。
- IDEA冻结SHA-256：`5ccf61e564d321cc32fd4c92d193f1802020aae0d1c55e189d3067d0387fb0db`。
- 大包：`2_1_preop_scans_coreg.zip`，3,164,137,716 bytes，官方MD5 `0d24a3a2c76bc6fa9b5b6468679676b1`；只提取冻结152个`t1c`成员。
- mask包：`3_1_preop_masks.zip`，1,982,029 bytes，官方MD5 `cca9e9da6d7cc4a06e0c31bebb86b592`；SHA-256 `51daf938a519d7b0d965940dfbe2852a29f0269cb8fd73467d9fe58a31e2056e`。
- 证据身份：public external-to-project、institution-disjoint/no-known-overlap、not hidden blind、not patient-ID-proven disjoint。

## 冻结模型与流水线

- Full972 random-init voxel-KD epoch168 checkpoint SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`：同一次循环输出identity与固定Flip8。
- GCSAM fit864 inner-selected epoch184 checkpoint SHA-256 `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`：identity。
- 预处理：nibabel closest canonical RAS；image/mask native shape与affine严格匹配（shape exact、max affine error≤`1e-5`）；raw image有限且有正值；raw label近整数、union只允许`{0,1,2,3}`；WT=`label>0`；SimpleITK全局线程1；canonical N4 shrink2/原迭代；image-only nonzero bbox；source-nonzero z-score；image trilinear、label nearest缩放到128³。
- Full views固定顺序：`(),(-1,),(-2,),(-2,-1),(-3,),(-3,-1),(-3,-2),(-3,-2,-1)`；各raw logit逆翻，float32等权平均。GCSAM只`()`。
- BF16 autocast、batch8、阈值`logit>0`；Point为raw-logit argmax；无SAM、模型ensemble、连通域、形态学或其他后处理。
- 模型顺序固定Full后GCSAM；总forward=152×(8+1)=`1368`。

## 指标、统计与解释门

- 绝对：Full identity、Full Flip8、GCSAM identity的Scan Dice、Subject Dice、Point hit；每人一scan，Scan=Subject但两列仍保留。
- 配对：Full Flip8−identity；GCSAM identity−Full identity。报告差值、improve/worse/tie与Point差。
- 10,000次percentile bootstrap，seed基准`20261452`：Full identity `+0`、Full Flip8 `+10`、GCSAM `+20`、Flip8配对`+100`、GCSAM配对`+200`。每人一scan，按病例配对重采样。
- GCSAM与Flip8各自三档：delta>0且Scan CI lower>0=`supported_on_WAW`；delta>0且CI跨0=`positive_but_uncertain`；delta≤0=`not_supported_or_reversed`。这不是promotion gate。
- 所有结果无论输赢完整发布；不做跨cohort pooling、显著性合并或总体排名。

## 工程门与停止边界

1. 下载后大包size/官方MD5必须exact；zip成员basename与冻结152 T1c一一对应，不能有缺失/重复。
2. 首例只做array/geometry/N4/cache探针；全量cache完成后重验源文件、raw label union、finite/binary/nonempty、首例byte exact、cache hashes。
3. 在模型前冻结IDEA/PREREG/cohort/source/cache/checkpoint/evaluator/smoke SHA的`MODEL_PREACCESS.json`；synthetic smoke只运行9 forward，不读取WAW图像。
4. 任一门失败立即停止并归档失败边界；不改容差、病例、modality、N4、线程、阈值、view或模型。
5. 性能只允许一次；开始前写started receipt。结果不得触发WAW重跑、病例剔除、调参或旧cohort重访。

## 访问声明与成本

- 当前已访问：WAW公开metadata与全部152 masks用于来源/标签门；未访问coreg MRI大包、WAW模型性能、checkpoint。
- 本实验不访问internal test270、fixed validation108、outer、confirmation、UCSD75、Africa95、CPTAC33的图像/cache或模型；只在结果解释中引用已归档数值。
- 训练0；预计大包2.95 GiB、目标提取<1 GiB、cache约1.48 GiB；GPU正式1368 forward，预计峰值约5.1 GiB。
- checkpoint训练源仍是本地BraTS2024 Kaggle mirror，官方Synapse corrected-v2 parity未证明。
