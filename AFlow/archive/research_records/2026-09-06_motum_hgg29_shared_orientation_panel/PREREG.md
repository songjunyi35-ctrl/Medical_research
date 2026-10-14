# MOTUM-HGG29共享方向冻结面板：预注册

状态：冻结于shared cache、checkpoint和MOTUM性能访问之前。

- IDEA SHA `c9f90194e3eae68e3dc9c20fe7a658d77569c894af264a75cd3ade491244c50d`；cohort SHA `f49e8f888b0fce8b7f34da1f9eadb49039d10aaed8e0538008f215df1b953379`；source audit SHA `69d59768cd54ed526aa1cc8b0f66e31884ea4aaad171a1a16bc17c4488412932`。
- Bridge `RESULT.json` SHA `3d349094d9e7a7c690f6f5f5e6ee9080cd07616c247915a5f7c5d0afdc32d5a9`；旧失败 SHA `394f02162a19b46516deaee1051efa4b7bb821ab0c0a7d28cf6bb79d4e27baa3`。
- 每例native shape exact、affine≤`5e-7`、三`io_orientation` exact；只从image生成到RAS transform/shared affine，同一离散transform作用三arrays，0插值；转换arrays须等于各自independent canonical arrays。
- WT=`flair_seg_label1>0 OR t1ce_seg_label2>0`；29例固定、38 metastases排除；0 raw WT crop loss；N4 shrink2/[50]×4/threads1、image-positive bbox/z-score/128³。
- Full KD168 SHA `f7df2a…80d17b` identity+8 reflections；GCSAM184 SHA `acf2fce…016d0d` identity；batch8/BF16/threshold0/argmax Point，无SAM/后处理/ensemble，总261 forwards。
- Bootstrap10k base `20261481`，absolute偏移0/10/20，paired偏移100/200；三档解释与IDEA一致，无promotion。
- 首例probe→全29 cache/byte gates→6+ tests→9-forward synthetic smoke→MODEL_PREACCESS→唯一formal。任何门失败停止；started后绝不重跑。
- Public external-to-project、三中国医院institution-disjoint/no patient-ID bridge、非hidden blind；21/24层与5.5–6.75mm grid caveat。正式不访问internal/validation/outer/confirmation/UCSD/Africa/CPTAC/WAW。
