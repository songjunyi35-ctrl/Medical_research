# ICCV 2019：真正使用图像区域强度的作者代码

仅论文与静态代码调研，不训练、不调用AFlow模型、不读取病例像素。本次作者仓库commit为 `d3be2fe6adc05843aa3d5dde2733ca43b3e5b149`，抓取SHA见 [来源收据](ICCV2019_SOURCES.json)。

## 身份与研究对象

论文是 Shir Gur、Lior Wolf、Lior Golgher、Pablo Blinder 的 **Unsupervised Microvascular Image Segmentation Using an Active Contours Mimicking Neural Network**。ICCV 即 International Conference on Computer Vision，国际计算机视觉大会。[CVF会议正式页面](https://openaccess.thecvf.com/content_ICCV_2019/html/Gur_Unsupervised_Microvascular_Image_Segmentation_Using_an_Active_Contours_Mimicking_Neural_ICCV_2019_paper.html)确认2019会议身份；[论文原稿](https://arxiv.org/abs/1908.01373)明确链接 [shirgur/UMIS](https://github.com/shirgur/UMIS)，仓库也声明官方实现。因此是可以明确归属作者的顶会代码。

对象是显微血管，不是BraTS肿瘤；利用血管在荧光成像中偏亮的特点，无标签训练。没有医学三维通用提示模型或点击提示；不能把它的结果当作当前T1ce整体肿瘤任务保证。

## 代码到底怎么算

[train_unsup.py 115–144行](https://github.com/shirgur/UMIS/blob/d3be2fe6adc05843aa3d5dde2733ca43b3e5b149/train_unsup.py#L115)拿网络软分割计算图像前景/背景均值，分母加1e-8。均值和图像梯度先计算，再对软分割做三对形态池化，最后调用loss反传。没有用真实标签区域均值；没有detach切断均值梯度。训练循环虽然数据加载器返回标签，但通过 `(data, _)` 丢弃标签。

用重新命名的简写表示：

- s：sigmoid输出的前景概率；I：输入强度。
- a=sum(s)，b=sum(1-s)。
- mu_F=sum(I*s)/(a+epsilon)。
- mu_B=sum(I*(1-s))/(b+epsilon)。

[分割网络126–129行](https://github.com/shirgur/UMIS/blob/d3be2fe6adc05843aa3d5dde2733ca43b3e5b149/networks/segmentation.py#L126)显式使用sigmoid；不是先将输出硬阈值化为二值mask后求梯度。[形态池化8–24行](https://github.com/shirgur/UMIS/blob/d3be2fe6adc05843aa3d5dde2733ca43b3e5b149/morphpoollayer.py#L8)提供自定义CUDA backward，说明形态平滑也在训练梯度链中；本次未编译或执行该扩展验证数值梯度。

## 原论文分支 EL 与后加分支 LS 不能混称

[README](https://github.com/shirgur/UMIS/blob/d3be2fe6adc05843aa3d5dde2733ca43b3e5b149/README.md)明确区分原论文 Euler–Lagrange（EL，基于能量变化的驱动力）和后加的 level-set（LS，水平集）分支；当前parser默认LS，不等于2019原论文原样运行。

[loss_functions.py 5–44行](https://github.com/shirgur/UMIS/blob/d3be2fe6adc05843aa3d5dde2733ca43b3e5b149/networks/loss_functions.py#L5)是EL分支：像素到前景均值的平方差减去到背景均值的平方差，分别乘1与2，再乘软分割的空间梯度幅值，构成驱动力F。F<0的位置鼓励分到前景，F>0的位置鼓励分到背景，分组使用指数惩罚。完整loss还加亮暗排序、熵、方差、重建和面积项；代码权重分别包括1e-2、1e-3、1e-3、1e-6以及面积系数。其分组选择不可微，但被选中的值及软分割仍留在梯度路径；不能笼统说所有操作处处光滑。

[loss_functions.py 47–62行](https://github.com/shirgur/UMIS/blob/d3be2fe6adc05843aa3d5dde2733ca43b3e5b149/networks/loss_functions.py#L47)是LS分支。核心能量的数学简写是：

`E = sum[ s*(I-mu_F)^2 + (1-s)*(I-mu_B)^2 ]`

仓库将该能量乘0.01，再加前景应较亮的排序项、前景均值项及面积项。它同时检查前景与背景中每个像素是否符合所在区域，而不是把整块区域压成一个均值后仅比较两个数字。

## 归一化：论文与仓库默认需分别说

论文说明先均值/标准差归一化，再把范围伸展到0–1。[数据代码9–13行](https://github.com/shirgur/UMIS/blob/d3be2fe6adc05843aa3d5dde2733ca43b3e5b149/data/datasets.py#L9)实现这两个步骤；实际沿每个z层的x/y轴计算。训练数据入口317–321行只有启用 `--range-norm` 才执行两步，未启用默认只做按层标准化。论文默认、仓库当前命令默认不完全一样，强度尺度不能略过。

## 与当前灰度项的区别

当前AFlow灰度项为 `(mu_pred - mu_GT)^2`，只要求预测前景平均灰度接近真实肿瘤平均灰度。UMIS是无监督图像区域分割：使用预测前景及其补集的均值，以像素残差形成分割能量/驱动力，再施加空间与防退化约束。两者都可以用软区域统计，证明“soft mask本身不是偏离公开研究的异常做法”；但公式、监督来源、亮暗假设、组织与成像域都不同，不能当成师兄MSCL的官方实现或直接替换补丁。

直觉上，当前项检查“这一篮东西的平均颜色像不像肿瘤”；UMIS还逐个检查“这个位置更像前景还是背景”，再约束轮廓。后者也有错误分组和假设失效风险，并不因发表顶会就保证对T1ce整体肿瘤有效。

## 公开代码可运行性的限制

静态核对已看到：train_unsup.py 第137行调用level_set时没有传args，但函数签名要求args；两种loss都读取args.lmd_area，而训练parser未定义该参数。因此不能称这份当前master已原样跑通。找到官方代码与完成可运行复现是两件事；本次未修补、未运行、不把后加LS误当原论文EL。
