# 复用开发角色：冻结完成

用户授权后，从原972 scans/444人train池中排除旧confirmation的53人/108 scans，留下864 scans/391人。按授权前声明的SHA256患者排序（seed2026090701），固定313人fit/78人development，实际为684/180 scans；同患者全部扫描同组，组间与protected confirmation/fixed-validation/test患者均无交叉。

这是reused development，旧开发经历依然存在，重分折不使数据变成fresh。原有manifest、数据、checkpoint和报告均保留；没有更改测试角色。拆分只读取metadata，无影像数组、模型或新性能。

- [授权](AUTHORIZATION.md)
- [机器划分收据](SPLIT.json)
- fit SHA：`75504df75fa734723945a9f07d76696fdcca89b8c29563dc0ad02a2560718d61`
- development SHA：`53885b7e300f915d81eecb115b82123e25330b7da422d3ff0b889b0489a2cdac`
- 首个批准范围内实验：[训练期切除腔辅助监督](../2026-09-07_structured_background_aux/IDEA.md)。
