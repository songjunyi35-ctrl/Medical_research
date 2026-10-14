# ReMIND量化感知DICOM几何冻结协议

状态：2026-09-05 UTC，在任何T1ce/SEG `pixel_array`读取前冻结。已有T1ce header访问与两份失败probe完整披露。

## 冻结输入

- DICOM SEG下载收据SHA256=`270ab28554f76f11142f6956efbfcb8cd54c659dbe5318e29c082c335880c4f9`，三个series重复内容门全部通过。
- Fixed-1e-5失败结果保留：192 unique positions/SOP/instances；gap最大中位误差0.000860437mm；未读pixel arrays。
- pydicom3.0.1 wheel/extracted manifest SHA保持`db32f78…`/`49e2b56e…`；不新增decoder。
- 只复用`/tmp/aflow-remind-dicomseg-probe-20260905`，不发网络请求、不打开旧NRRD。

## DS区间算法

1. 只接受普通十进制DS字符串，可含正负号和小数点，不接受NaN/Inf。对字符串s解析Decimal中心c；若小数位d，半宽h=`0.5×10^-d`；无小数点则h=0.5。科学计数法若出现直接失败，不临时定义。
2. 两DS值的区间相交当且仅当`max(c1-h1,c2-h2)≤min(c1+h1,c2+h2)`。
3. 相邻source位置中心差向量v；误差分量上界为两端半宽之和，L2总界b。距离区间固定为`[max(0,||v||-b), ||v||+b]`。每片SpacingBetweenSlices必须字符串exact相同且解析为1；191个距离区间都须包含1，lower>0。
4. 192片ImageOrientationPatient六个原始字符串逐项exact；InstanceNumber 1–192/SOP unique。排序仍用orientation中心法向投影，投影必须严格单调；不要求派生gap固定1e-5。

## SEG门

- 沿用前一PREREG的SEG身份、BINARY/BitsAllocated1/单segment、frame shape、ReferencedSegmentNumber1和唯一Source SOP门。
- 每frame PlanePositionSequence三个DS分别须与所引source ImagePositionPatient区间相交；不使用一个统一mm容差。
- Shared PixelMeasures PixelSpacing两个DS分别与source PixelSpacing区间相交；Rows/Columns exact。若字段缺失/科学计数/多引用/压缩handler缺失即失败。
- 按SOP重建192-slice二值mask，两个正体素>0，tumor⊂cerebrum。未引用slice补0，不重采样。

## 访问/停止

- 只在全部source interval门通过后读取T1ce和两个SEG pixel arrays；报告解码前后failure boundary。
- 0 checkpoint/GPU/model/metric，其他42例无像素。全门通过也只授权另写input bridge。
- 任一失败不改算法/区间/病例/decoder，新增CPU墙钟≤10分钟。
