# BraTS-Africa TCIA package948来源审计计划

状态：冻结在package payload、NIfTI数组、标签值和模型访问之前。

目的：确认TCIA公开的BraTS-Africa Version 1能否提供一个与本地BraTS2024七个欧美来源站机构独立、带专家标注的T1CE/WT外部队列。当前只读取TCIA页面、小型官方datainfo和Faspex package目录metadata。

固定源：TCIA collection DOI `10.7937/V8H6-8X67`，public-link package ID `948`，页面标称1.6GB。递归列出全部官方路径/类型/大小并保存哈希清单；不得下载任何package payload。

审计问题：

1. package内实际病例数和文件总数；
2. 每例是否有唯一T1CE和唯一专家seg；
3. 是否可从路径明确区分2023 train/validation/test或公开post-challenge角色；
4. 是否存在缺失/重复/非正大小文件；
5. 只有完整清单通过后，才能另立固定首例source probe；本审计不授权标签解码、cache或模型。

停止：package身份不等于948、目录分页不完整、病例/T1CE/seg配对不唯一、角色仍隐藏或来源需受控访问时，均在payload前停止。不能用Kaggle/第三方镜像替代。

一手来源：[TCIA BraTS-Africa](https://www.cancerimagingarchive.net/collection/brats-africa/)、[2026 challenge paper](https://doi.org/10.1093/noajnl/vdag082)、[2025 dataset paper](https://doi.org/10.1148/ryai.240528)。
