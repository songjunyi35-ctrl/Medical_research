# 顶会作者代码如何使用图像灰度：三类目标必须分清

2026-09-17。用户要求查顶会官方仓库的真实实现。本次并行核读论文、作者仓库及loss调用链，仅文献/静态代码工作；没有新训练、新患者推理或新精度结果。原SAM灰度训练仍停止161/200。

## 回答

找到可核对的一手实现，但没有证据表明“预测区域均值−真值区域均值的MSE，权重1”是统一的顶会标准做法。下面三个顶会案例解决的目标不同，不能互相冒充；另附一个更直接的区域灰度损失期刊实现，明确它不是顶会。

| 已核实论文及作者代码 | 实际信号 | 怎么用 | 与本地版本的关系 |
|---|---|---|---|
| ICCV2019：Unsupervised Microvascular Image Segmentation Using an Active Contours Mimicking Neural Network；[UMIS官方库](https://github.com/shirgur/UMIS) | 输入影像强度及soft前/背景概率，不依赖GT区域 | 前后景均值形成逐像素残差/轮廓驱动力，加空间、防退化项 | 使用灰度，但不是mu_pred对齐mu_GT的监督损失 |
| ECCV2018：On Regularized Losses for Weakly-supervised CNN Segmentation；[作者rloss库](https://github.com/meng-tang/rloss) | 0–255 RGB颜色、坐标、softmax分割概率 | 颜色与空间相似的像素形成亲和矩阵，惩罚不一致分类 | 图像亲和度约束，非区域均值MSE，原实现2D RGB |
| CVPR2019：Learning Active Contour Models for Medical Image Segmentation；[作者库](https://github.com/xuuuuuuchen/Active-Contour-Loss) | y_true二值标签与预测概率 | 标签区域项+预测边界长度项 | 虽叫主动轮廓，但官方loss不接收MRI图像，不能当灰度损失 |
| TIP期刊：Mumford–Shah Loss Functional for Image Segmentation With Deep Learning；[官方库](https://github.com/boahK/TIP_MumfordShahLoss) | 输入图像与各类别软概率 | 每类概率加权均值→逐像素残差求和→边界平滑→监督项 | 直接的灰度区域能量参考，但非顶会、非师兄方法 |

会议身份一手证据：[ICCV2019页面](https://openaccess.thecvf.com/content_ICCV_2019/html/Gur_Unsupervised_Microvascular_Image_Segmentation_Using_an_Active_Contours_Mimicking_Neural_ICCV_2019_paper.html)、[ECCV2018页面](https://openaccess.thecvf.com/content_ECCV_2018/html/Meng_Tang_On_Regularized_Losses_ECCV_2018_paper.html)、[CVPR2019原论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf)。这些是明确标明年份的已核实例，不宣称最新方法或同BraTS条件结果。

## 1. ICCV2019：软前景/背景的区域强度与轮廓

直觉：不仅看整个预测区域的平均颜色，还检查每个位置更像前景还是背景，同时约束轮廓和防止全部分成同一类。

[train_unsup.py#L115](https://github.com/shirgur/UMIS/blob/d3be2fe6adc05843aa3d5dde2733ca43b3e5b149/train_unsup.py#L115)中，网络用sigmoid输出软概率p，分别计算mu_F=sum(I*p)/(sum(p)+1e-8)、mu_B=sum(I*(1-p))/(sum(1-p)+1e-8)，均值没有detach。原论文EL（Euler–Lagrange，基于能量变化的驱动力）分支使用像素到两个均值的残差形成轮廓推动力，还配合形态池化、亮暗排序等项。

必须区别：仓库后来增加的LS（level-set，水平集）分支，才可简写为 `sum[p*(I-mu_F)^2+(1-p)*(I-mu_B)^2]`，外加其他约束；不能把这个后加分支称2019论文原式。当前master默认LS；原论文是EL。

对象是荧光显微血管，包含前景偏亮的假设；不是T1ce整体肿瘤。论文说z-score后伸展0–1，代码是否伸展取决于range-norm开关。静态检查还发现当前公开分支有args参数接口遗漏，未宣称原样跑通。[完整核对与源码位置](ICCV2019_AUDIT.md)。

## 2. ECCV2018：颜色与位置相似的像素，预测也应相近

[官方DenseCRFLoss.py](https://github.com/meng-tang/rloss/blob/1caa759e568db2c7209ab73e73ac039ea3d7101c/pytorch/pytorch-deeplab_v3_plus/DenseCRFLoss.py#L21)把反归一化0–255图像的颜色和xy坐标送入双边过滤，形成颜色/空间亲和度；输入分类结果是softmax概率。有效区域掩膜排除padding。目标为每类别负二次型加总并除batch数，不是逐像素平均；不需要GT肿瘤区域均值。

官方示例的sigma_rgb=15、sigma_xy=100、weight=2e-9、scale_factor=0.5都与其RGB尺度/像素归约绑定。缩小图像时空间带宽同步缩小；不能把15或2e-9直接套在标准化MRI。

forward转CPU numpy只是调用高效过滤库，自定义backward显式给出概率梯度−2AS/N并乘有效区掩膜；图像与带宽不反传。因此看到detach/numpy不能直接断言它“没梯度”。这是有源码可核的特例。原Caffe版与后续PyTorch版均已核读。[完整核对](CVPR2018_AUDIT.md)；文件名保留检索起点，但正文明确该具体案例是ECCV2018，Normalized Cut Loss才是同作者CVPR2018的另一篇。

## 3. CVPR2019：名字像区域灰度，官方实现实际吃标签

[作者loss文件](https://github.com/xuuuuuuchen/Active-Contour-Loss/blob/master/Active-Contour-Loss.py)的输入只有y_true和y_pred，没有MRI影像。区域项取预测前景概率乘(y_true−1)²、预测背景概率乘(y_true−0)²，并加预测概率图差分形成的边界长度。对二值标签，区域项可化为平均[p(1-y)+(1-p)y]。

因此它用标签约束前景/背景和轮廓，不是比较真实MRI区域亮度均值。官方示例中区域相关lambda与mu均为1，边界项有1e-8数值稳定项，代码硬写256×256；不能把这两个1当成本项目灰度权重1的文献依据。作者README还提示早期训练可能不稳定，建议预训练模型；这不等于它能解决当前掉点。

## 4. 更直接的图像灰度能量：TIP作者Mumford–Shah实现（期刊）

直觉：把图像划成若干区域，每个区域内的像素应接近该区域的代表亮度，同时轮廓不要过于破碎。

在[models/loss.py#L14](https://github.com/boahK/TIP_MumfordShahLoss/blob/master/models/loss.py#L14)，对每个类别概率p_k和输入图像I：

`mu_k = sum(I*p_k)/sum(p_k)`

`L_region = sum_k sum_pixels p_k*(I-mu_k)^2`

它计算的是区域内部逐像素残差，不是(mu_pred−mu_GT)²。空间梯度项对预测概率差分取绝对值并求和。上游网络末端Softmax2d，调用时概率夹到[1e-10,1]，没有先做硬二值化；均值与区域能量有正常反向链。

已沿[真实调用位置](https://github.com/boahK/TIP_MumfordShahLoss/blob/master/models/unet_model.py#L88)确认loss.py中名为target的参数传入的是**self.real_A图像**，而不是self.real_B标签。监督标签另用于交叉熵。总组合是监督项+lambda_A×(区域能量+0.001×空间梯度项)。[配套LiTS训练脚本](https://github.com/boahK/TIP_MumfordShahLoss/blob/master/scripts/LiTS_train_unet.sh#L26)实际给lambda_A=1e-6；parser默认10是另一层默认，不能只看默认不看运行配置。

图像处理器按图像通道减最小值，再除以移位后最大值，目标是0–1尺度；区域项对空间、类别和batch求和，因此权重随分辨率、batch、强度尺度和归约变化。它是旧版2D肝脏CT实现，不是三维MRI直接可用配方。本次只读取，未声称软件版本或边缘情况已跑通。

## 对当前项目真正有用的判断

- soft概率并非我们独有的错误：上述实际图像区域统计同样使用可微软分区。
- 几种目标不能混称：区域均值对齐、区域内逐像素偏差、颜色/空间亲和度、标签轮廓正则不是同一个loss。
- 必须连输入强度尺度、前后景定义、空区处理、空间/batch归约和总权重一起核对。官方示例1、1e-6、2e-9不能脱离各自实现比较大小。
- 这几份代码通常还结合空间、边界或其他防退化约束；本地仅用一个前景均值约束的辅助项不包含这些信息，但这不直接证明哪个目标更适合本任务。
- 把任何一个替换进SAM都属于新方法，不是自动修复师兄公式。当前没有选择或实施新loss，没有开训练，也没有用已揭示test挑选候选。

当前仍保留真实结果：灰度50轮test270/119 best5 Scan0.566424，161轮0.515107，旧交互基线0.594226。以上文献没有同输入、同病例、同权限的直接成绩对比，不能宣布能超越基线。历史全自动0.757270总体/0.752003单次前向属于另一模型口径。

## 材料

[ICCV作者代码报告](ICCV2019_AUDIT.md) · [ICCV源码SHA](ICCV2019_SOURCES.json) · [ECCV作者代码报告](CVPR2018_AUDIT.md) · [ECCV源码收据](rloss_sources/SOURCE_RECEIPT.json) · [TIP补充源码收据](TIP_EXTRA_SOURCES.json)。源码副本只用于静态核对，没有执行第三方训练代码。
