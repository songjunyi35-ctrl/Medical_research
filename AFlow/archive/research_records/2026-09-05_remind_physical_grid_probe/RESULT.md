# ReMIND物理网格映射：SEG内部spacing区间门失败

状态：当前ReMIND物理grid实现关闭，且本轮不再设计第五种ReMIND-002映射。Source量化区间通过后，两个SEG各自frame位置与声明SliceThickness的exact量化区间不一致；在SEG pixel arrays和任何resampling前停止。0模型/0指标。

## 结果

Source T1ce的191个量化感知gap继续全过。Manual tumor SEG为36 frames、声明SliceThickness1mm；35个相邻gap范围`0.999994205–1.000001863mm`、中位`1.000000160mm`，相对1层最大残差`5.7946e-6`。Automatic cerebrum为275 frames、声明0.3995816mm；274个gap范围`0.399576446–0.399591259mm`、中位`0.399579836mm`，除以声明厚度后的最大整数层残差`2.4173e-5`。

两者相邻frame都对应一层，不是稀疏frame跳片；失败来自DCMQI输出的有限精度position/orientation与声明thickness没有落在由各DS文本末位推导的严格区间。数值很小，但预注册不允许事后引入统一容差或非均匀slice映射，故不读取SEG `pixel_array`、不做nearest resample。

## 解释和停止边界

物理坐标映射是一般正确方向，但当前“DS文本量化区间足以解释全部转换误差”的实现不成立。要继续需从DICOM转换数值规范或独立样例预先推出额外数值误差模型；本轮已连续看到同一首例header，不应再以这些值设计第五套门。因此关闭ReMIND当前线，保留43例metadata候选供未来独立窗口，不运行模型。

## 成本和访问

- 3项物理grid合同；0新网络、0GPU、0checkpoint、0forward。
- T1ce pixel array已由前一量化probe读取；两个SEG仍仅header，SEG pixel array=false；旧NRRD array=false。
- nearest mapping未执行、mapped mask不存在；其他42例无像素。

证据：[`IDEA.md`](IDEA.md)、[`PREREG.md`](PREREG.md)、[`AMENDMENT_01.md`](AMENDMENT_01.md)、[`IMPLEMENTATION.json`](IMPLEMENTATION.json)、[来源总审计](../../reviews/2026-09-05_remind_source_audit/RESULT.md)。
