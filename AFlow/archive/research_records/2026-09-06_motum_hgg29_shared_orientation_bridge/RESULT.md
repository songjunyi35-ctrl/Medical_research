# MOTUM-HGG29共享离散方向桥：结果

状态：✅ input-only双进程门通过；0插值、0cache、0checkpoint、0模型/性能。

两个独立Python进程各自处理29例。29/29 native三文件shape exact、image-mask affine≤`5e-7`、离散orientation arrays exact；从image计算一次到RAS的轴翻转/置换，同样应用到两个mask。共享转换后的image/flair-mask/t1ce-mask分别与各文件独立`as_closest_canonical`数组bit-exact，说明像素/标签没有改变；下游只统一使用image导出的shared affine，避免微小header小数在大索引flip中放大。

Copy A/B的29例case payload（transform、shared affine SHA、三array SHA、WT union SHA）完全一致。这个桥解决旧实验的`1.4488e-5`canonical affine假失败，但不解决21/24层、5.5–6.75mm厚层或半自动标注限制。

决定：旧MOTUM性能预注册仍关闭，不修改。可另立`shared-orientation`性能IDEA/PREREG，完整重绑cache/evaluator/preaccess后一次性评测；桥本身不授权任何结果驱动选择。

证据：[七问](IDEA.md)、[预注册](PREREG.md)、[机器结果](RESULT.json) SHA-256 `3d349094d9e7a7c690f6f5f5e6ee9080cd07616c247915a5f7c5d0afdc32d5a9`、[copy A](BRIDGE_COPY_A.json) SHA `c54465eb68447b4ebd708a4ed34a7e8452522283bbd53c1048f1738910f178cd`、[copy B](BRIDGE_COPY_B.json) SHA `968586671b6c9c56613e8a80c86acc501d7268f03b0cefd256db9302e4887fcd`。
