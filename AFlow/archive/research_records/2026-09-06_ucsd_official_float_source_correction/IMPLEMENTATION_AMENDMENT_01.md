# Implementation amendment 01: frozen source-status literal

时间：2026-09-06 UTC  
边界：正式cache前、GPU smoke前、UCSD模型forward前；cache不存在，模型/指标访问为0。

首次全量metadata/source-SHA绑定检查在读取任何NIfTI或运行模型前停止。原因是新builder把冻结来源比较文件的状态合同误写成自拟的`official_images_differ_labels_exact`，而实际已哈希归档的值为`completed_source_identity_audit`。病例数、标签exact、几何和0/75图像exact仍由后续独立字段检查，不依赖状态名暗示。

允许修复仅一项：把状态字符串改成冻结文件中的实际literal。病例、文件、哈希、预处理、模型、阈值、视图、指标、bootstrap和无性能晋级规则均不变。修复后须重跑聚焦测试和全量75映射/150文件SHA绑定；若仍失败，继续在模型前停止。
