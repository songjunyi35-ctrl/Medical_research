# OpenNeuro-GBM334冻结checkpoint流式面板：预注册

状态：冻结于任何图像/标签payload、checkpoint加载、GPU和性能访问之前。

## 冻结来源、队列与角色

- 数据集：OpenNeuro `ds007045`，快照`1.0.1`，snapshot id=`ds007045:1.0.1`，DOI `10.18112/openneuro.ds007045.v1.0.1`，CC0。
- [来源审计](../../reviews/2026-09-06_openneuro_ds007045_source_audit/RESULT.md) SHA-256 `9de549c734efeaeae136b19db353ff468cc55f94463afb5fb1f6c26c947e985f`；[机器对象清单](../../reviews/2026-09-06_openneuro_ds007045_source_audit/SOURCE_INVENTORY.json) SHA-256 `4bba11fb31913b18f44428087d9daf9dcd8f49590577ea11b715ddbcbbbf66eb`；规范化334对manifest SHA-256 `8342256890d3b6a8ebf408ff081a84867f76fd816da0e97e76f98b41e1cce6cb`。
- IDEA SHA-256 `bfda8f45c1c5b981a13676335ba14672320283a6eea8171ac61a733808376e4f`。
- 337 participant IDs中，像素前固定排除`sub-BO199`/`sub-NSK07`（缺segmentation）和`sub-BO97`（缺冻结processing T1-CE）；保留严格交集334 subjects=334 scans。`sub-BO8`唯一mask名为`MNI-152`，仍须与影像过严格geometry门。
- 八站BA/BO/FO/JS/KA/ME/NSK/TO=`3/170/9/11/80/7/40/14`。相对BraTS2024七站只称institution-disjoint/no-known-overlap，不称patient-ID-proven disjoint。
- 证据角色：一次性public external-to-project，not hidden blind；不用于checkpoint、阈值、视图、融合、病例或预处理选择。

## 冻结输入与流式实现

- 每例versionId URL和git-annex SHA/size已逐项冻结。影像固定为`derivatives/processing/<subject>/<subject>_t1ce_MNI152_brain.nii.gz`：发布者已MNI152线性配准、skull-strip，但未做其N4/z-score。标签固定为`derivatives/segmentation`最终expert-reviewed/refined mask。
- 源总量2,177,416,379 bytes，当前磁盘不足以常驻完整source/cache。只允许逐例下载到固定临时目录、SHA验证、处理/推理后覆盖；不生成334例float32 cache。每例源SHA和输入门详情写入raw。
- canonical：nibabel closest canonical RAS；shape exact、max affine error≤`1e-5`；图像/标签finite；label近整数且子集`{0,1,2,3}`；WT=`label>0`且非空；SimpleITK全局线程1；N4 shrink2/迭代`50,50,50,50`；image-positive bbox必须包含100% WT；非零z-score；image trilinear、label nearest到128³。
- 首例按排序后的`sub-BA2`。首例完成后同样固定对象与函数再跑一次，128³ image/label byte SHA必须exact；不访问其他333例。

## 冻结模型与输出

- Full972 random-init voxel knowledge distillation（体素知识蒸馏，KD）epoch168：`workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth`，SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`。
- GCSAM fit864 inner-selected epoch184：`workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth`，SHA-256 `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`。
- Full views固定为`(),(-1,),(-2,),(-2,-1),(-3,),(-3,-1),(-3,-2),(-3,-2,-1)`；各raw logits逆翻后float32等权平均。第一个identity视图同时给Full identity，因此Full共8 forwards。GCSAM仅identity 1 forward；总计334×9=`3006`。
- BF16 autocast，单例batch；阈值`logit>0`；Point为identity/融合后的raw-logit argmax；无Segment Anything Model（SAM）、模型ensemble、连通域、形态学或其他后处理。

## 指标、统计和解释门

- 绝对：Full identity、Full Flip8、GCSAM identity的Scan Dice、Subject Dice、Point hit；一人一scan所以Scan=Subject但两项均保存。
- 配对：Full Flip8−identity、GCSAM identity−Full identity；报告mean delta、improve/worse/tie和Point差。
- 10,000次病例percentile bootstrap，base seed=`20261640`：三个绝对依次`+0/+10/+20`，Flip8配对`+100`，GCSAM配对`+200`。
- 三档解释：delta>0且配对Scan CI lower>0=`supported_on_OpenNeuro_GBM334`；delta>0且CI跨0=`positive_but_uncertain`；delta≤0=`not_supported_or_reversed`。无性能promotion。
- 所有成功工程门后的结果无论正负均完整归档并更新独立Leaderboard分榜；不跨cohort pool或按绝对Dice混排。

## 工程门、失败边界与访问

1. 首例下载必须与冻结size/SHA exact；label union子集、geometry、零WT crop loss、128³有限/二值/非空与重复byte exact全过。
2. Synthetic smoke只用随机张量跑Full八视图+GCSAM一次，验证shape/device/内存；不读OpenNeuro病例。
3. 正式前冻结IDEA/PREREG/source inventory/checkpoints/evaluator与smoke哈希到`MODEL_PREACCESS.json`，再写started receipt。
4. 正式逐例先下载并验证该例两文件，再跑预处理工程门，再推理；任一失败立即停止并记录已下载、已预处理和已访问性能的精确数量。网络瞬时失败只可对同一versionId/SHA对象重试。
5. 不得丢失败病例、扩GT crop、换raw/N4/z-score variant、放宽geometry、改阈值/视图/模型。若334例未全部完成，则不得报告完整队列分数或进入榜单。
6. 本面板不访问internal test270、fixed validation、outer、confirmation、UCSD/Africa/CPTAC/WAW/MOTUM影像/cache或模型；旧数字只从归档报告引用。

## 成本

- 训练0；单卡；预计峰值约5–6 GiB。
- 网络读取约2.03 GiB；常驻新增仅raw、收据、报告；单例临时通常<10 MiB。
- 正式3006 forwards；Full Flip8参数不增但部署延迟约8×，identity/GCSAM各1×。
- 继承checkpoint仍保留本地BraTS2024 Kaggle mirror与官方Synapse corrected-v2字节parity未证明的限制。
