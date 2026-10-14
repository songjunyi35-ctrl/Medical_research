# RHUH发布态Otsu-N4输入桥接：独立重建门失败

状态：当前byte-reproducible实现关闭。80例首轮工程门均通过，但固定首3例的最终image SHA在独立重建中0/3 exact，因此整体`all_gates_passed=false`。没有重跑、改线程、放宽门或删病例；不影响正在训练的BraTS Otsu候选及其已冻结cache。

## 结果与失败边界

TCIA package684固定后两个时点80 scans/40 subjects逐文件SHA重验后，背景解码、Otsu200拟合mask、N4完整正支持、image-only bbox、源正值population z-score和128³输出均单次通过。边界背景众数最小占比`0.983648`；Otsu拟合体素范围`1,075,619–1,618,201`；标准化最大绝对均值`5.61e-11`、population std最大误差`1.83e-10`。

失败发生在独立重建：首3例brain mask SHA为3/3 exact、Otsu fit-mask SHA为3/3 exact，但最终float32 image SHA为0/3 exact。现有报告没有保存两次完整float数组，因此不能诚实量化差值大小，也不能把“hash不同”写成准确率必然不同。按预注册，byte exact是全门，故不补跑诊断、不把近似相等替代exact。

## 解释

“Otsu能给RHUH产生finite且支持稳定的单次输入”得到支持；“当前跨进程Otsu-N4实现能byte-exact重建”不成立。fit-mask和输出mask都一致，把失败定位在浮点校正/插值值而非病例身份、前景选择或几何支持，但证据不足以进一步定位SimpleITK N4还是后续浮点算子。

这只关闭RHUH发布态的当前exact重建实现，不否定Otsu拟合mask的一般方法，也不改变BraTS train cache：该cache已在自身冻结环境下构建并重建通过，Stage A仍按原协议继续。即使Otsu候选未来性能过门，本轮也不据此下载RHUH标签或运行模型；需要新的、独立预注册且不以RHUH分数调参的工程路线才能重新打开。

## 成本和访问

- 2项合成合同；正式input-only CPU墙钟`634.269s`，8 worker；0 GPU、0模型forward。
- 不建持久图像cache；仅KB级报告。
- RHUH标签未下载/未读取，checkpoint未加载，Dice/Point为0；fixed validation、outer、confirmation、test270、UCSD75均无访问。

证据：[`INPUT_QC.json`](INPUT_QC.json)、[`RESULT.json`](RESULT.json)、[`PREREG.md`](PREREG.md)、[`IMPLEMENTATION.json`](IMPLEMENTATION.json)。
