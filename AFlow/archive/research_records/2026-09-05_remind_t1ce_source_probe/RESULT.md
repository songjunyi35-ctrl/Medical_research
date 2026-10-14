# ReMIND-002来源探针：双传输完整性门失败，像素QC未运行

状态：当前NRRD双传输probe实现关闭。固定首例的官方文件已部分下载，但`all_download_gates_passed=false`；按预注册没有打开DICOM/NRRD数组、没有几何/标签域QC、没有模型forward或Dice/Point。其他42个候选没有像素访问。

## 已确认的来源价值

像素前的官方metadata链得到43名候选：病理精确为19 Glioblastoma、15 Astrocytoma、9 Oligodendroglioma；每人有唯一preop T1 postcontrast series与直接参考T1 postcontrast的manual whole-tumor NRRD，34/43另有同T1ce automatic cerebrum。ReMIND来自Brigham and Women’s Hospital单一机构，不在BraTS2024七个贡献站，许可CC BY 4.0；一手论文说明肿瘤由神外fellow手工画并由attending neurosurgeon复核。

## 失败边界

官方`.sums`的356/356 MD5全是空文件值`d41d8cd…`，因此修订为“两次独立官方传输的本地SHA/bytes exact”。ReMIND-002第一轮manual tumor/cerebrum分别为4,747/1,947,370 bytes，SHA256=`2fb1bf3d…`/`63ddccbd…`，符合浏览API size；第二轮Aspera在文件前报`Unable to connect via SSH`，两文件均不存在，双传输门失败。不重试、换病例或把第一轮单独放行。

NBIA ZIP有193个非目录成员而预期实例数192。离线只读ZIP目录证明额外成员是2,786-byte `LICENSE`；192个DICOM总未压缩bytes为21,660,158，与官方series FileSize exact。实现错误地把所有ZIP成员及LICENSE bytes计入DICOM门，故还产生一个假失败；但即使修正该计数，独立NRRD第二传输仍真实失败，整体结论不变。

## 解释与范围

该结果否定的是“当前一次性NRRD双传输probe已完成”，不是ReMIND数据或43例候选本身。失败发生在第二轮网络连接，未触达标签值/几何/输入预处理。原始DICOM和第一轮NRRD保留在`/tmp/aflow-remind-probe-20260905`，没有静默删除；只可作为失败证据，不能继续本probe QC。

若继续ReMIND，应提出独立且预先冻结的新来源实现，例如使用TCIA同collection的DICOM SEG而非重试本NRRD流程；必须保持ReMIND-002与43例定义固定、记录已经发生的像素下载，并仍在模型前解决未去颅骨输入问题。

## 成本与访问

- 下载ZIP8,896,341 bytes（含192 DICOM+LICENSE）；NRRD首轮1,952,117 bytes，第二轮0。
- 5项合成/合同测试通过，0 GPU、0checkpoint、0模型forward、0指标。
- 已下载但未解码：ReMIND-002 DICOM pixels、manual tumor NRRD、automatic cerebrum NRRD；segmentation voxel access=false。
- fixed validation、outer、confirmation、test270、UCSD75、RHUH模型均未访问。

证据：[`COHORT_CANDIDATES.json`](COHORT_CANDIDATES.json)、[`PROBE_PREACCESS.json`](PROBE_PREACCESS.json)、[`AMENDMENT_01.md`](AMENDMENT_01.md)、[`DOWNLOAD_RECEIPT.json`](DOWNLOAD_RECEIPT.json)、[`IMPLEMENTATION.json`](IMPLEMENTATION.json)。
