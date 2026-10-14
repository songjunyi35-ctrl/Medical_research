# MOTUM-HGG29冻结checkpoint面板：模型前关闭

状态：❌ 首例canonical geometry门失败；0 cache、0 checkpoint、0 model forward、0性能。

## 发生了什么

官方双传输、HGG29选择、29×3 source native shape/affine和标签语义均通过。但预注册要求图像与两个mask各自运行`closest canonical RAS`后，shape相同且affine最大误差≤`1e-5`。首例MOTUM-HGG-0001 native三文件shape均192×256×21、axis均LPS、image-mask affine误差仅`2.384e-7`；独立canonical后shape/axis仍相同，affine误差却放大到`1.4488e-5`，超过冻结门，故在N4/输出写入前停止。

只读扩展到29例发现：29/29的离散orientation arrays完全相同、canonical shape相同；native最大误差仍`2.384e-7`，但2/29在独立canonical后超过`1e-5`，最大同为`1.4488e-5`。原因是极小方向余弦差在191/255索引翻转的canonical affine平移计算中被放大，不是mask数组需要插值的证据。

## 决定

当前实验按预注册关闭，不放宽`1e-5`、不丢2例、不加载模型。这个实现失败不证明MOTUM无法评测：可另立input-only bridge，在native shape/affine更严格且三文件离散orientation完全相同的前提下，从图像计算一次离散flip/permutation并原样用于两个mask；它不插值GT，也不使用性能。如果该桥跨进程29/29 exact，再新建独立性能预注册，而不是复活本实验。

## 成本与访问

训练/GPU/model/cache均0；已完成双官方984,434,960-byte source下载及29×3 arrays来源审计。MOTUM性能、任何旧cohort均未访问。

证据：[七问](IDEA.md)、[预注册](PREREG.md)、[失败机器记录](CACHE_PROBE_FAILURE.json)、[来源审计](../../reviews/2026-09-06_motum_hgg_source_audit/RESULT.md)。
