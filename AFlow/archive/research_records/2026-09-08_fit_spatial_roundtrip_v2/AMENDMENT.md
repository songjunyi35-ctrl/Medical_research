# V2：统一原生体素距离单位，不猜毫米、不排病例

V1因代码要求image和label均显式mm，在第3个fit病例停止。已保存首例独立过程与全量前2行；4进程预取可能还访问了其他fit标签，旧版本未完整记录启动事件，不能宣称只读了2例。所有任务仅来自fit684，0开发/test/模型访问；V1脚本、PREACCESS、部分RAW和FAILURE保留不改。

只读全684 header复核：单位组合(mm,mm)501、(mm,unknown)123、(unknown,mm)46、(unknown,unknown)14；canonical shape全部一致、affine最大差0，数值轴步长全部(1,1,1)。未知单位不等于标签坏，但不能默认解释成毫米。证据为V1目录HEADER_UNITS_AUDIT.json。

修正只针对计量单位和访问记录：全队列表面距离在原生体素索引坐标计算，spacing=(1,1,1)，改名HD95_native_voxels、mean_surface_*_native_voxels及surface_dice_one_native_voxel。另原样记录image/label单位和affine轴长度。没有选择哪个病例可报告，也不把14例无声明的距离写成mm。

Dice、体积、image-only crop、nearest-exact128、两还原算子/阈值和26邻接组件追踪完全不改。旧core及其测试保持原哈希，新入口只调整单位适配，输出到新目录；PREACCESS同时绑定V1失败记录、header审计、本修订及新入口/测试。每worker独立访问日志记录header开始、标签数组读取开始、完成，明确预取边界。

这是工程/计量纠正，不是根据Dice挑算子或放宽性能门；仍不加载模型，不访问dev/旧队列，不更改训练方案或标签。未知毫米单位不再阻塞单位无关的保真诊断。
