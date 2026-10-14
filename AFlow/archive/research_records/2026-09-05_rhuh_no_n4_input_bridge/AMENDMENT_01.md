# Amendment 01：RHUH一次性外部确认的执行顺序与时间门

状态：2026-09-05 UTC，RHUH标签未下载、checkpoint未加载、模型forward/Dice/Point均为0时冻结。不修改IDEA/PREREG的病例、输入、模型、视图、阈值或成功门。

## 为什么需要补充

输入桥接80/80通过后，标签官方inventory metadata也80/80 exact；但no-N4 E/F尚未揭示，Full972 checkpoint尚不存在。为避免资格出现后临时决定下载/评测顺序，本修订固定动态checkpoint SHA如何进入preaccess，以及30小时窗口尾部的停止条件。

## 固定顺序

1. 联合Stage A完成归档后，从`RAW_INNER.json`重新计算no-N4每fold×identity/Flip8及跨fold门；不能只信任保存的authorization布尔值。
2. 仅no-N4全门通过，且条件Full972 runner完成no-N4 fixed168训练与配对工程门，才读取其`TRAIN_DONE.json`并重算checkpoint SHA。
3. 在任何RHUH标签传输前写不可覆盖`EXTERNAL_PREACCESS.json`，冻结checkpoint/report/input-QC/label-inventory/官方`.sums`哈希、80/40清单、全指标、bootstrap和访问状态。
4. 开始外部阶段时距离30小时截止`2026-09-06T14:02:39Z`必须≥2小时；不足则只保留ready状态，不下载标签、不启动不可能完整归档的评测。
5. 先用合成128³图像做checkpoint/8镜像GPU smoke，不打开RHUH标签；通过后才从TCIA package684只下载冻结的80份segmentation，逐bytes/MD5/SHA核对。
6. 构建`/tmp`外部cache：图像只用已通过的no-N4桥接；标签geometry必须与图像一致，整数值域必须是`{0,1,2,3,4}`子集，Whole Tumor固定为`label>0`，80例都必须有正WT。标签不用于bbox、病例选择或输入变换。
7. 全部工程门后只运行一次固定Flip8评测。报告128³栅格Scan Dice、每人两时点等权Subject Dice、80例Point hit；患者簇percentile bootstrap10,000次、seed20261361。绝对描述门仍为Scan≥0.70、Subject≥0.70、Point≥72/80。
8. 无论正负，保存raw/per-case/preaccess/checkpoint/cache/report SHA，更新handoff/catalog及独立RHUH external leaderboard section。禁止回流调模型、输入、阈值、view、病例或重测。

## 不变的访问边界

不访问fixed validation、outer、internal confirmation、test270或UCSD75。RHUH是新的外部cohort，不与内部或UCSD排名混排；40 subjects可能给出较宽CI，点估计过门不等于相对其他模型显著。
