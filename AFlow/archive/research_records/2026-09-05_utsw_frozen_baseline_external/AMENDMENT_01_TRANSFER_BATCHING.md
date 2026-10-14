# Amendment 01：固定100路径批传输与missing比较修复

冻结时间：2026-09-05T23:54:06Z，第二次source执行前；0 payload/NIfTI/cache/model。

首次双copy执行中，Aspera在两个会话启动时均以`too many source files`/returncode1退出，724个payload全部不存在。总状态正确为failed并阻止后续，但描述性`files_copy_exact=724`错误地把两边`None==None`算作相同。原实现、报告和失败目录保留。

唯一工程变化：

- 724个冻结相对路径按字典序切成固定100路径最大批次，即每copy 8批、共16个Aspera连接；不搜索客户端上限。
- 每copy仍使用一次独立公开授权；每批独立请求只含该批冻结路径的transfer spec，并使用同一`--src-base`目录保留规则。
- 写入全新`/tmp/aflow-utsw-frozen-baseline-source-v2-20260905`，不覆盖失败现场。
- `copy_exact`只有在A/B都存在、两边官方size exact、两边SHA非空且bytes/SHA相同才为true。

cohort、724路径、双copy、官方size、模型、cache、统计、成功门和“任何NIfTI前724/724全过”不变。代价只是连接数从2增至16。

