# MOTUM-HGG29共享离散方向桥：预注册

状态：冻结于bridge worker执行前；0模型/性能。

- IDEA SHA `6645975309475795eac231944b48f0365d0334c76b2f72b27bbab887ca2a7ec9`。
- Cohort SHA `f49e8f888b0fce8b7f34da1f9eadb49039d10aaed8e0538008f215df1b953379`；source audit SHA `69d59768cd54ed526aa1cc8b0f66e31884ea4aaad171a1a16bc17c4488412932`。
- 旧失败 SHA `394f02162a19b46516deaee1051efa4b7bb821ab0c0a7d28cf6bb79d4e27baa3`；旧实验关闭、不改旧门。
- 每例native三文件shape必须exact，image-mask affine max≤`5e-7`，三个`io_orientation` exact。
- RAS目标orientation固定；只由image native affine计算`ornt_transform`与`inv_ornt_aff`。同一transform用`apply_orientation`作用于image/flair-mask/t1ce-mask；0插值。
- 转换后每个数组必须与该文件独立`as_closest_canonical`的数组bit-exact；双mask仍按各文件`>0`合并，FLAIR值域⊆0/1、T1ce⊆0/1/2、WT非空。
- 两个独立spawn Python进程，各自产生29例case ID、transform、shared affine SHA和三个array SHA；整份cases必须exact相同。
- 输出固定`BRIDGE_COPY_A.json`、`BRIDGE_COPY_B.json`、`RESULT.json`；任一已存在拒绝重复。
- 无N4/cache/checkpoint/GPU/model/performance；通过只作为新性能协议的输入资格，不改变极厚层/半自动标注/患者桥限制。
