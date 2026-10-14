# Amendment 01：物理grid接受合法科学记数DS

状态：2026-09-05 UTC，在物理grid实现和任何SEG pixel_array访问前冻结。已知SEG header的SliceThickness/PixelSpacing使用`1.000000e+00`、`4.000000e-01`等合法DICOM DS；若机械继承前一“科学记数直接失败”规则，会在已知格式上产生无信息失败。

本方法的DS interval改为标准Decimal解析普通/科学十进制，拒绝NaN/Inf。中心为Decimal值；最末有效十进制位的quantum由`Decimal(text).as_tuple().exponent`给出，半宽=`0.5×10^exponent`。例如`1.000000e+00` exponent=-6、half=5e-7；`4.000000e-01` exponent=-7、half=5e-8。位置/spacing的区间包含、距离界计算不变。

这不是从SEG像素或模型结果调容差，而是按已访问header的合法表示补全parser。病例、series、nearest、axis顺序、5%体积门、tumor⊂cerebrum与0模型边界不变。
