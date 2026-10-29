# ROOT旧cache只读盘点与源文件保护

A未删除、移动或备份任何载荷，没有模型解码、患者数组解码、新前向或GPU操作。实际删除由总控统一执行；当前native108、RACE270和训练worker/tmp缓存不在本清单内。

在读取忽略文件规则之外的实际文件后，发现以下旧派生载荷；不是仅据目录或index计数：

| 对象 | 实际文件 | 逻辑字节 | 说明 |
|---|---:|---:|---|
| 已关闭684/180支线的archive runtime/cache | 864 NPZ | 1,436,679,288 | 旧预处理影像/标签；原始source、disabled builder、CACHE_INDEX及模型须保留。B负责逐SHA与活动FD核验。 |
| nnU-Net ResEnc525旧test_logits | 270 NPZ | 1,675,690,134 | 连续margin软输出，旧融合原子缓存；不是模型或最终JSON/掩膜报告。B负责逐SHA与活动FD核验。 |
| smoke_single_repo_refiner | 2 F16 | 8,388,608 | 不加入“已确认可重建”辅助白名单，见下。 |

原972/108/270的持久输入缓存、public334_relay及其他历史输入缓存目前只有index/删除或迁移标记，images/labels实际载荷不存在，**可再释放载荷按0计**。不能把2026-10-21曾释放的14.21875GiB或public334旧迁移收据再次计为空间。当前worker上存在的原生数组也不能冒充ROOT持久盘可清理项。

保护清单已完成：

- [WEIGHTS_BEFORE.json](WEIGHTS_BEFORE.json)：ROOT全部193个`.pth/.pt/.ckpt/.safetensors`文件，合计11,139,192,401字节，逐文件SHA256且hash前后元数据稳定。包括旧关闭支线、Hugging Face发布载荷和初始化权重，不因方法终止而删除。
- [SOURCE_METADATA_BEFORE.json](SOURCE_METADATA_BEFORE.json)：1097个受保护文件，合计9,074,429,826字节；覆盖`data/**`、`artifacts/checkpoints/**`以及原始MRI和源压缩包扩展。保存路径、bytes、mtime、inode、模式等，不读取临床数组内容。
- 所有历史最终分数、逐例统计、已选掩膜、研究报告、冻结文件、构建源码和索引均保留。搜索分数矩阵和历史predictions本次也保守列为研究证据，不作为cache释放数。

smoke的两份实际F16均匹配各自index的历史SHA，生成脚本仍在。但index指定的旧SAM权重`42f96a0a…`当前路径已不存在，上游GT-crop images/labels也已删除，故不能承诺现可直接精确复建。仅保留builder并不等于模型依赖齐备。[具体依赖与SHA](SMOKE_CACHE_REBUILD_AUDIT.json)。是否清除此8MiB由总控判断并如实登记依赖缺失，A不执行删除。

旧684输入builder为保留的`closed_material/scripts/sammed3d_graymean_pair.py.disabled`及`build_t1ce_wt_official15_cache.py`，原`pair_technical/CACHE_INDEX.json`保留每份NPZ/输入/标签SHA。nnU-Net连续margin生成入口为`nnunet_resenc_20261013/finalize_v3.py`，旧`TEST_INFERENCE_JOURNAL.json`保留270文件SHA；固定525权重及原始source仍保护。重建模型输出需要重新运行网络，清理许可不代表现在重新评测或重启旧coordinator。

删除后可由总控运行[protect_inventory.py](protect_inventory.py)的`--phase after`，会重新逐SHA核对全部旧模型bytes和路径，并比较所有旧source元数据；只写after清单，不执行删除。A的before总清单与脚本SHA见[SUMMARY_BEFORE.json](SUMMARY_BEFORE.json)。
