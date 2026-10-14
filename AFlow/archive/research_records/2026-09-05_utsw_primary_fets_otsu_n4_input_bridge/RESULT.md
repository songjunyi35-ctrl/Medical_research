# UTSW主FeTS图Otsu-N4输入桥接结果

## 结论

固定3例input-only门通过。BT0001/2/3的两个独立Python进程、SimpleITK线程1输出image/label/geometry与Otsu fit-mask均3/3 exact；相对no-N4和canonical正值maskN4，两组的label/geometry都保持3/3 exact，Otsu image均3/3不同。

这保留了“若只有Otsu Stage A晋级，可再设计UTSW Otsu-vs-canonical外部确认”的工程可能；没有模型成绩，也不授权全362。

## 唯一变化与结果

主FeTS图、全网格`r`标签、all-positive→WT、shrink2/四级50次、RAS、bbox/z-score/128³均固定；相对canonical只把N4拟合mask从全部正值换为Otsu200自动前景。

- fit mask体素：1,338,299/1,499,055/1,431,390；完整正支持为1,370,681/1,519,442/1,470,146。
- mask只用于拟合，输出完整正支持保留。
- 两进程child SHA：`eda0243f…`/`d0f19d62…`。
- Otsu输出image SHA：`35968659…`/`f87f4523…`/`a50a98a2…`。
- 所有冻结比较门均3/3通过。

## 直觉、代价和限制

支持“RHUH Otsu重建失败不等于UTSW直接非负主图也失败”的工程直觉；只说明确定性与接线，不说明Otsu分割更准。两个进程总371.3秒CPU，0GPU/0forward，复用已下载约56MB；相关5项合同通过。

当前Otsu E/F仍未开卷，其余359个人工病例未访问。若Otsu最终未通过，停止在本输入资产；若Otsu通过，仍须另立全362 source/cache/checkpoint/配对评测协议，不得直接复用no-N4队列改arm。

## 证据

- [七问/最终填写](IDEA.md)
- [预注册](PREREG.md)
- [实现冻结](IMPLEMENTATION.json)
- [预访问](PREACCESS.json)
- [copy A](COPY_A_QC.json)
- [copy B](COPY_B_QC.json)
- [INPUT_QC](INPUT_QC.json)，SHA `55b7331695fcdb80723d68c4ed0377d275cab547d2b8f2f52b2a8448fd58a82c`
- [机器结果](RESULT.json)

