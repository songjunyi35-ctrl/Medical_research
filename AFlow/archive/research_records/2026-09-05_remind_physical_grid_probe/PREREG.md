# ReMIND DICOM SEG物理网格映射冻结协议

状态：2026-09-05 UTC，在任何SEG pixel_array访问前冻结。T1ce pixel已在量化source probe读取；两个SEG仅header已读。

## 冻结输入

- DICOM SEG下载收据SHA256=`270ab28554f76f11142f6956efbfcb8cd54c659dbe5318e29c082c335880c4f9`。
- 量化source probe结果：191/191 source interval门通过；随后因Rows/Columns不同和无SOP引用停止。SEG pixel arrays未访问。
- pydicom3.0.1/SimpleITK2.5.5；临时依赖hash不变；不打开旧NRRD或请求网络。

## 物理grid合同

1. 对source和每个SEG，ImageOrientationPatient必须所有frame/instance原始字符串exact；法向=`cross(first3,last3)`且norm有限正。按法向投影PlanePosition排序，严格单调。
2. 使用前一方法相同DS半量化区间。每个相邻frame欧氏距离区间必须包含Shared PixelMeasures SliceThickness声明中心值；source同样包含SpacingBetweenSlices=1。区间下界>0。
3. pydicom pixel_array必须shape=`frames,Rows,Columns`、值域{0,1}。构造SimpleITK时array轴为z,y,x；spacing x=`PixelSpacing[1]`、y=`PixelSpacing[0]`、z=`SliceThickness`；direction的三列依次为IOP first3、IOP last3、cross；origin=排序首frame position。
4. Source reference用相同规则和已按projection排序的192 DICOM slices构建；source z spacing=1。通过SimpleITK identity+nearest-neighbor、default0、uint8把SEG映射到source。
5. 原SEG positive voxel index bbox的8角物理点，转换到source continuous index后每轴必须位于`[-0.5,size-0.5]`。若orientation/axis接反，该门应失败，禁止尝试另一排列。
6. 原/映射物理体积=`positive voxels×spacing product`，相对误差`abs(mapped-original)/original≤0.05`；两个mapped mask二值非空，manual tumor必须是automatic cerebrum子集。
7. 保存mapped mask SHA、体积/geometry/bbox；不保存/训练模型，不选择N4/brain extraction。

## 停止与访问

- 所有header/interval门后才首次读取两个SEG pixel arrays。0 checkpoint/GPU/forward/metric；其他42例无像素。
- 任一失败关闭当前ReMIND路线；不改5%/nearest/axis顺序/病例，不以旧NRRD或模型分数调映射。
- CPU≤10分钟、新持久archive≤20MB，mapped数组只记录hash不落cache。
