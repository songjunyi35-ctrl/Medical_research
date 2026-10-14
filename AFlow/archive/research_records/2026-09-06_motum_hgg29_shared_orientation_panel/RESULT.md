# MOTUM-HGG29共享方向冻结面板：模型前关闭

状态：❌ shared-orientation桥通过，但首例0-WT-crop-loss门失败；0 cache/checkpoint/model/performance。

共享方向接口成功解决旧canonical affine误差，首例三arrays仍无插值对齐。但冻结canonical N4流水线把T1ce≤0置零，并只用图像正值支持求brain bbox。MOTUM-HGG-0001的raw WT为21,007 voxels，正值图像bbox内只有20,115，丢892（4.246%）；违反预注册0丢失门，因此在N4输出/首例receipt前停止。

只读量化29例：仅0001有丢失，28例为0，总丢失892。不能因“只有一例”事后排除，也不能用GT扩框、把正值改非零、切no-N4或放宽门。MOTUM processed T1ce带负值，说明当前positive-support N4/crop接口与该来源不完全兼容；共享坐标桥解决几何，不解决强度支持。

这只否定当前shared-orientation+positive-support canonical实现，不证明MOTUM或所有背景解码不可能。未来若有不看GT、来自发布者定义的brain/background mask或重新训练且过独立证据的兼容输入方法，可另立协议；当前不继续自创source-tuned offset。

成本：双source与29×3来源审计已发生；新增cache/GPU/model/performance=0。证据：[七问](IDEA.md)、[预注册](PREREG.md)、[失败记录](CACHE_PROBE_FAILURE.json)、[通过的坐标桥](../2026-09-06_motum_hgg29_shared_orientation_bridge/RESULT.md)。
