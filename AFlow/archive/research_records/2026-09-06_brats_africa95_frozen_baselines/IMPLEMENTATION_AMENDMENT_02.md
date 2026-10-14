# Implementation amendment 02: match N4 thread count and finalize completed memmap

全95例worker都已完成，随后首例byte-exact门失败，故没有index/cache receipt/model。只读重算确认95例memmap均finite/binary/non空、标签exact、文件大小exact；唯一失败是worker image与首例probe最大差`4.482269e-5`。Worker明确将SimpleITK设为1线程，首例probe遗漏该设置并使用默认线程；probe重复运行又与原probe SHA一致。

不放宽byte门，不删除/覆盖失败cache。允许两项纯工程修复：

1. V2首例probe在任何N4前固定PyTorch/SimpleITK各1线程；原probe保留。
2. 若V2输出与现有worker memmap首例image/label byte exact，独立finalizer重新读取95例已访问源，复核源SHA、geometry、finite、integer、允许标签域、WT非空和全部memmap，再为现有完整memmap写index/receipt；不会重跑N4或修改tensor。

V2仍不exact则关闭cache；不得容差放行。模型/checkpoint、数据、all-positive WT、N4参数、crop、128³、阈值、视图、指标和bootstrap均不变。
