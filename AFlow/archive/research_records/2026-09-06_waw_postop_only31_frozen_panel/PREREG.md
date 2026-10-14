# WAW术后only-patient31冻结面板：预注册

状态：冻结于磁盘回收、postop大包、cache、checkpoint和性能访问之前。

- IDEA SHA `4f21bcd4babbf229b683c84c9bb1a3992ec6cf3da90203beaaf0b9187d100cfc`；source audit SHA `ca7d67892e42ae4d193f87ebdd983fb79314091b84d2e448021a71da18e0fe3d`；cohort SHA `965b2fe0f331d782dbb6bb136d6089c1181419a9a75dbd8f198cbeff33284180`。
- 31 scans=subjects；与已评WAW preop152 patient overlap0；只取postop-only、numeric days、每人最早day/exam，0–22天。4 blank-only patients、17 later numeric scans、57 preop-overlap patients固定排除。
- 大包`2_2_postop_scans_coreg.zip` 3,321,955,416 bytes、official MD5 `ab9da3eb2a3aff73b8f3b95e01508c1c`；只提取cohort31的`t1c`。
- Image/label closest canonical RAS、native shape exact/affine≤`1e-5`；image finite/positive；raw labels近整数⊆`{0,1,2,3,4}`；WT=`{1,2,3}`、RC4 background、31 WT nonempty。
- N4 shrink2/[50]×4/threads1、image-positive bbox、0 WT crop loss、positive z-score、128³ trilinear/nearest；首例与全cache byte exact。
- Full KD168 SHA `f7df2a…80d17b` identity+固定8 reflections；GCSAM184 SHA `acf2fce…016d0d` identity。Batch8/BF16/threshold0/argmax，无SAM/postprocessing/model ensemble；279 forwards。
- Bootstrap10k base seed `20261531`：Full/Flip8/GCSAM absolute +0/+10/+20，Flip8/GCSAM paired +100/+200。
- 三档解释：positive+CI lower>0=`supported_on_WAW_postop31`；positive+CI cross0=`positive_but_uncertain`；delta≤0=`not_supported_or_reversed`。无promotion，三绝对/两配对全报。
- Source/cache/smoke/checkpoint/preaccess任一失败即停；不换时间点/病例/阈值/view/N4/crop。Started后不可重跑。
- 正式只访问WAW-postop31；不访问preop WAW、internal、validation、outer、confirmation、UCSD、Africa、CPTAC。Public早期post-resection、同WAW机构但新患者、institution-disjoint/no BraTS patient bridge、非hidden blind。
- 磁盘：删除前精确核已归档WAW preop selected extraction与MOTUM copy_b；保留WAW preop官方zip、MOTUM primary zip及全部raw/results/hashes。删除仅为可重建/冗余副本，不改变证据。
