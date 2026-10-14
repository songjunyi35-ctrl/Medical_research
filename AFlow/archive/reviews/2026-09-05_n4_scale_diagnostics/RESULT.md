# N4数值原因隔离：float32偏移往返损伤近零信号，随后被N4放大

状态：train-only诊断完成；没有标签、模型、E/F或旧cohort预测指标访问；正在运行的no-N4实验不变。

## 结论

原先“换强度单位使N4严重变化”的解释过于笼统。固定3份train MRI的因果对照表明：

1. 原输入重复两次，3/3完整128³张量逐字节相同，迭代轨迹也完全相同。
2. 精确乘0.5/2，最大差≤0.000131；同病例迭代次数相同。
3. 直接除以相同sigma，最大差≤0.000122；最敏感第三例仅0.000003815。
4. 经过float32全图z-score、再减负背景恢复后，第三例近零正信号的最大相对恢复误差22.085%，对数误差0.19955；最终张量max/mean差0.057679/0.002245，最后一级迭代从11次变为10次。
5. 把这个编码/恢复往返放在float64中，最后才转float32，三例结果均与直接缩放条件逐字节相同；第三例相对原输入重新降至max0.000003815。

这支持：负背景浮点编码中的大数相减损失了接近零的小信号精度，N4的对数/迭代流程把这类输入扰动放大。没有证据把某个内部步骤单独认定为软件bug；不能归因于执行随机性，也不能将float32偏移误差等同于纯比例单位变化。

三例小于1的正值体素数为354、0、405。第一例float32往返的最大相对误差2.605%，完整链max差0.000139；第二例相对误差约3e-7、完整链max约2.4e-6；第三例最明显。这是描述性train图机制诊断，不是分割准确率或统计显著性结果。

## 版本排错

环境准确版本为SimpleITK2.5.5 / ITK5.4.6，和官方SuperBuild一致。最初怀疑来自v5.4.0中最高直方图档的缺失分支，但当前v5.4.6已经添加`else correctedPixel=E.back()`；该旧版本解释被排除。没有安装/编译/修改ITK，也不把已修复的问题当新成果。

## 后续边界

背景解码在原正值mask-N4管线的等价性失败仍保留，不放宽门，不运行原RHUH评测。新no-N4试验继续按原冻结方法验证是否需要额外校正。另一个在任何E/F成绩前注册的Otsu拟合mask候选，单独检验可否保留校正同时降低近零噪声影响；这是新mask因素，不是恢复旧失败协议或调整其容差。

两轮诊断分别155.95和155.94秒CPU；0训练GPU、0模型forward、无新大缓存。没有读取分割标签；采用原先固定首3份train图。无法从这些输入数值差异直接推出Dice更好或更差。

证据：[第一步预案](PLAN.md)、[原输入重复/倍数缩放](RAW_DIAGNOSTIC.json)、[第二步预案](PLAN_02.md)、[精度因果隔离](RAW_OFFSET_ROUNDTRIP.json)、[Otsu独立预注册](../../research_records/2026-09-05_n4_otsu_clean_replication/PREREG.md)。

来源：[ITK5.4.6实现](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.6/Modules/Filtering/BiasCorrection/include/itkN4BiasFieldCorrectionImageFilter.hxx)、[SimpleITK2.5.5依赖](https://github.com/SimpleITK/SimpleITK/blob/v2.5.5/SuperBuild/External_ITK.cmake)、[官方N4说明](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1N4BiasFieldCorrectionImageFilter.html)。
