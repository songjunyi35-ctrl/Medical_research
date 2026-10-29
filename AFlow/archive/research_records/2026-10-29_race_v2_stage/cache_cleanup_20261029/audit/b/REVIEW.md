# B旧缓存清理审计（只审计，未删除）

状态passed，精确白名单1134个NPZ，逻辑3,112,369,422 bytes，allocated 3,114,659,840 bytes。所有文件常规非symlink、nlink=1、独立inode；完整SHA逐项与保留历史INDEX/JOURNAL相同。

|候选|实际文件数|逻辑bytes|allocated bytes|重建与索引证据|
|---|---:|---:|---:|---|
|关闭684/180预处理cache|864|1,436,679,288|1,438,445,568|closed_material/pair_technical/CACHE_INDEX.json；sammed3d_graymean_pair.py.disabled缓存入口与build_t1ce_wt_official15_cache.py|
|旧nnU-Net270 continuous margin cache|270|1,675,690,134|1,676,214,272|TEST_INFERENCE_JOURNAL.json；nnunet_resenc_20261013/finalize_v3.py|

读取zip目录元数据核验：灰度cache恰image.npy+label.npy，NN缓存恰margin.npy；没有解码临床数组、GT或加载网络。所有builder、index/journal和逐项SHA都写入PAYLOAD_SHA_AUDIT。第二项root已明确纳入新用户“旧缓存保留脚本”删除范围，最终JSON/掩膜/权重依旧保留。

head默认沙箱/proc受限，已在工具批准的只读升级后核验702个fd目标、33份maps，候选命中0；同UID应用仅sshd PID2011268不可读。1154121普通UID与sudo只读扫描互补，候选命中0；普通UID例外PID8243为短暂sshd且后续已退出。root基础设施服务与少量受保护进程仍有OS权限例外，完整例外列表保留，**不声称对不可读系统进程做过完整扫描**。AFlow已知应用没有候选打开/映射命中。

root删除前应再次核验此1134项的路径/bytes/SHA/nlink与当前进程使用情况，按白名单只unlink这些NPZ。不得递归删目录、碰.git、任何原始source/权重/最终报告/掩膜或当前1154259训练、1154121 108/RACE270 scratch。这里只产生本地文本审计；未转移payload，跨目的地备份拒绝边界未变。

[逐文件精确白名单与SHA](PAYLOAD_SHA_AUDIT.json) · [passed收据](PASSED_RECEIPT.json)。
