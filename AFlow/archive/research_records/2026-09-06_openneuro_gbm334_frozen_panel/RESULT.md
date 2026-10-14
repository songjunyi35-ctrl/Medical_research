# OpenNeuro-GBM334冻结checkpoint流式面板：输入门负结果

状态：canonical image-positive crop输入门失败，已在0外部模型/0性能处关闭；不进入Leaderboard。

## 初学者结论

这份2026年公开数据本身很有价值：八家医院、337名术前原发胶质母细胞瘤，标签全部经神经放射科医师复核。项目在任何像素前冻结了334个“未N4的MNI T1-CE+最终标签”配对，并逐例真正下载、验SHA、做N4和128³预处理。

问题出在项目固定的“只按图像正值区域裁剪”规则：`sub-BO118`有4个、`sub-KA86`有6个真实WT标签体素落在图像正值框外。数量很小，但预注册门明确要求0丢失；像考试前发现答题纸会切掉哪怕几个正确答案，也不能看见后临时把纸放大。因此334例面板没有加载checkpoint处理任何外部病例，也没有Dice/Point分数。

## 什么变了，什么没变

- 唯一候选变化是OpenNeuro `ds007045:1.0.1`的334例新队列。
- 模型、两个checkpoint、WT=`{1,2,3}`、canonical RAS、单线程N4、image-positive crop、非零z-score、128³、阈值0、八镜像、融合和统计计划均未更改。
- 预检读取全部334对版本化源对象，但不持久化源或cache；12个CPU进程并行。Checkpoint只在随机合成张量smoke中加载过，绝未与OpenNeuro数据同用。

## 工程结果

- 337个participant IDs中，版本化对象树严格交集334；BO199/NSK07缺mask、BO97缺冻结variant T1-CE，在像素前排除。
- 334对source size/SHA全部exact；332例通过完整输入门。
- 全部可解析标签union为`{0,1,2,3}`；shape/affine门均过；首例BA2两次独立N4/128³ image与label byte exact。
- `sub-BO118`：shape`182×218×182`、affine error0、raw WT348,742，crop外4体素，占`0.001147%`。
- `sub-KA86`：shape`182×218×182`、affine error0、raw WT97,700，crop外6体素，占`0.006141%`。
- 两失败都不是网络、checksum、未知标签、空WT或geometry问题；是冻结图像支持框与最终标签的真实不一致。

## 成功/失败门与决定

预注册要求334/334的source SHA、geometry、label、非空WT和零WT crop loss全过，才允许生成`MODEL_PREACCESS.json`。实际`all_zero_wt_crop_loss=false`且只有332/334完成，所以门失败。

决定：关闭当前exact canonical面板；不排除两例、不把标签裁掉、不用GT扩框、不试1/2/更多voxel padding、不换raw/N4/z-score variant、不放宽“0丢失”门，也不报告332例性能。该结果只否定“现成canonical image-positive crop可无损直接接这334例”的实现，不否定OpenNeuro数据集或所有可能的独立全FOV模型。未来若发布者修正brain support/标签，或有在新train-only证据上预先通过的全FOV模型，可另立新协议。

## 代价与访问

- 训练0；全量输入预检墙钟2,908.6秒（48.48分钟）、12 workers，流式读取2,177,416,379 bytes；常驻source/cache 0 bytes。
- Synthetic smoke为9 forwards、峰值706,991,104 bytes；它只用随机张量。
- 外部病例checkpoint/model forwards=`0`；OpenNeuro性能、internal test270、fixed validation、outer、confirmation及其他外部性能均未访问。
- 第一次首例探针异步收据稍晚显现，随后同一首例诊断重复启动；第二次只重读同一versionId/SHA首例，在发现收据已存在时退出。两次均0模型，不改变门或cohort。

## 证据

- [七问](IDEA.md)
- [预注册](PREREG.md)
- [机器结果](RESULT.json)，SHA-256 `21f90cbab1801d84162f3748a22de1690670fe47bd284a3c0be1eb1206d00a9a`
- [全334输入收据](INPUT_PREFLIGHT.json)，SHA-256 `c3e338d089527de46b83992ac871537f7702c48aa4f87eb78226531b8c60912c`
- [首例探针](FIRST_CASE_PROBE.json)，SHA-256 `e8bbc5d6eac0acac5921942ee929c11e1a1c5c6326aa15f8ba6e9c6369cc3c9b`
- [Synthetic smoke](SMOKE_RECEIPT.json)，SHA-256 `83bde5c3de06117a230b59365ea1bc5a07607e6878ad649e71f722e8e37e4ea5`
- [来源审计](../../reviews/2026-09-06_openneuro_ds007045_source_audit/RESULT.md)
- [版本化对象清单](../../reviews/2026-09-06_openneuro_ds007045_source_audit/SOURCE_INVENTORY.json)，SHA-256 `4bba11fb31913b18f44428087d9daf9dcd8f49590577ea11b715ddbcbbbf66eb`
- [Full checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth)，SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`
- [GCSAM checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth)，SHA-256 `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`
