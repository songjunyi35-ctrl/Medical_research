# Experiment idea: ReMIND DICOM SEG物理网格映射

Status: preregistered; SEG pixel arrays still unseen

## 1. 现在的问题是什么？

ReMIND-002三条DICOM series的下载复现全部通过，source切片的量化感知1mm区间也191/191通过；但manual tumor SEG是49×32×36、automatic cerebrum是415×335×275，源T1ce为256×216×192。两个SEG没有Source SOP逐帧引用，只提供PlanePosition、PlaneOrientation和PixelMeasures，因此SOP-only/Rows exact方法在SEG pixel前失败。还没有合法的43例标签→T1ce映射，模型为0 forward。

## 2. 一句话直觉

把不同分辨率的透明地图按真实经纬度叠到同一张底图，而不是强迫两张图使用相同格子编号。

## 3. 为什么选择这个idea？

- DICOM SEG正是用ImagePositionPatient、ImageOrientationPatient、PixelSpacing和SliceThickness定义像素的患者物理坐标；header已经证明这是裁剪/不同分辨率grid，不是错误series。
- ReMIND一手论文说明DCMQI以被分割MR作为reference转换SEG，且各preop图像在Brainlab中完成共配准；物理网格映射是发布格式要求。
- NRRD传输失败、固定1e-5 gap失败和SOP-only失败分别否定三种实现，没有读取SEG pixels，也没有否定标准physical-space nearest resampling。
- 仍固定ReMIND-002，不根据mask或模型换病例；使用nearest保持二值标签，不引入平滑/形态学。

## 4. 具体怎么做？

复用冻结DICOM下载。先沿用量化感知算法验证source与每个SEG内部相邻frame位置区间包含声明slice spacing，frame投影严格单调。按DICOM规则：数组column对应ImageOrientationPatient第一向量、row对应第二向量，frame法向为两者叉积；以首frame ImagePositionPatient为origin，PixelSpacing反序映射到SimpleITK x/y，声明slice spacing为z，构造两个二值SEG image。

用同样物理规则构造T1ce reference grid，以identity transform和nearest-neighbor把tumor/cerebrum各映射一次到T1ce grid。SEG正mask的物理bbox必须完全在source continuous-index边界`[-0.5,size-0.5]`内。比较原物理体积与映射后物理体积，误差≤5%；检查非空、二值及tumor⊂cerebrum。只报告源图非零组织在cerebrum外比例，不选择输入算法。

## 5. 唯一变化与固定项

- 唯一变化：SEG到T1ce从不适用的SOP-only直接拼接改为DICOM物理坐标上的一次nearest-neighbor映射。
- 固定不变：ReMIND-002、三个已复现series、pydicom/SimpleITK版本、DS量化区间、manual tumor/automatic cerebrum、5%体积门、无形态学/GT crop/模型。

## 6. 成功、失败和停止规则

- Primary工程门：source/两个SEG各自规则物理grid；orientation/position/spacing字段完整；所有positive bbox在source内；nearest输出二值非空；tumor/cerebrum物理体积相对误差各≤0.05；tumor⊂cerebrum。
- Promotion gate：全过才称首例DICOM SEG可映射，并只授权独立input bridge；不自动下载43例。
- Failure/stop rule：任一门失败不改5%、插值、origin/direction顺序、病例或mask，不用旧NRRD校准。
- 是否允许test：0模型/指标；不授权。

## 7. 成本与风险

- 网络/新增raw磁盘：0；复用已下载数据。
- CPU秒到分钟，内存预计<1GB；GPU/forward=0。
- 风险：DICOM行列方向实现接反会使物理bbox/volume/包含门失败；automatic cerebrum本身不是人工gold-standard，未来用它做skull strip会成为provided-mask assisted而非全自动。5%只控制栅格体积，不证明BraTS WT语义一致。

## 8. 最终结果（完成后填写）

- Validation/Test/CI：无模型指标。
- 是否支持直觉：待SEG pixel/物理grid QC。
- 下一步：全过才设计输入桥接；失败停止ReMIND当前线。
- 报告/哈希：待填写。
