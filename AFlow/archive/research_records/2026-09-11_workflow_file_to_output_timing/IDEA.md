# 文件读取到模型网格输出：效率边界检查

Status: engineering-only preregistration, independent of test success/failure.

## 1. 现在的问题是什么？

固定工作流的模型推理阶段已测3.961/2.652倍加速，但不含NIfTI读取、N4亮度校正、裁剪归一化缩放。历史972扫描缓存总build2534.37秒是并发构建吞吐，不能直接除病例数冒充单病例端到端延迟。

## 2. 一句话直觉

排队和准备占大半时间时，医生看得快四倍，整趟就诊不一定快四倍。必须测完整计时边界。

## 3. 为什么现在选它？

现有加速明确只含H2D到mask/point D2H，而原流水线含单线程N4校正。[SimpleITK N4官方示例](https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html)说明用偏置场校正缓慢变化的强度不均；本实验直接复用本项目hash绑定实现，不改其参数。旧阶段速度结果未被否定，只是尚未覆盖这部分成本。

## 4. 具体怎么做？

仅fit manifest前4例原始T1ce NIfTI→既有canonical读取→原N4 shrink2/4×50/非零mask→正值支持框crop→既有无标签归一化/resize128→固定模型full8或最终pruned→阈值0/argmax→CPU模型网格掩膜/点。保持模型常驻，不写临床NIfTI输出。

每个模型四例，每例分别从文件开始重做两次完整流程（full8/pruned），共16次。按case序号交替顺序，第二模型反转顺序；各模型先用原fit缓存前三例分别warmup两策略，预处理不复用计算结果。计时结束后核对重建128影像与原缓存逐字节一致、有限值、输出形状；不读取GT。不要求同一病例不同策略掩膜相等。

## 5. 唯一变化与固定项

只改视图集合full8→最终代表pruned，沿用已经冻结的两个native checkpoint与FINAL_WORKFLOWS；不按新test结果选模型/策略。原图路径、SHA、模型SHA、fit/cache SHA和源码在任何NIfTI数组加载前绑定。SimpleITK/OMP/MKL/OpenBLAS均1线程，batch1，同GPU顺序运行。不以新预处理替代原缓存。

## 6. 成功、失败和停止

工程正确性：16/16重建输入与既有缓存byte exact；若失败立即记录偏差并停止，不能修改容差或继续把输入不同的耗时当剪枝对照。报告每例/模型/策略读取+预处理秒、模型阶段秒、总秒，配对总耗时比的中位数。没有最低总加速晋级门；接近1也是重要边界，不能挑最好病例。单病例计时边界为文件读取至128³CPU输出，不包含模型加载、原生空间导出/文件写入、临床其他步骤，也不叫完整临床端到端。预先SHA读取会暖文件页缓存，不称冷启动I/O。无精度CI或显著性检验，四例是工程探针。

整个GPU/CPU执行阶段限30分钟；无test/开发/旧outer/confirmation/外部新访问，不把这里的时间回流改精度方法。

## 7. 成本与风险

0训练，16次原N4处理，模型前向=每模型(4+3)×(8+pruned视图数)，合计147次；3为每策略warmup例数。预计CPU数分钟至十几分钟，GPU短时，无大型新cache或checkpoint，仅JSON。主要风险：N4浮点/线程不一致、文件系统热缓存与小样本引起时延波动；据实限定。

## 8. 最终结果

待工程执行；不改变主研究或测试成败结论。
