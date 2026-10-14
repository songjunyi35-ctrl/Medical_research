# 顶会作者代码核读：ECCV 2018 DenseCRF loss

2026-09-17，只读论文及作者代码。未训练、未评测、未访问医学影像或模型权重。本报告沿任务约定文件名保存；正文明确会议纠正。

## 论文和官方身份

**On Regularized Losses for Weakly-supervised CNN Segmentation 是ECCV 2018，不是CVPR 2018。** [会议官方页面](https://openaccess.thecvf.com/content_ECCV_2018/html/Meng_Tang_On_Regularized_Losses_ECCV_2018_paper.html)确认作者与会议。相关的 **Normalized Cut Loss for Weakly-Supervised CNN Segmentation** 才是[CVPR 2018论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Tang_Normalized_Cut_Loss_CVPR_2018_paper.html)。这里深入核读前者DenseCRF代码，不能混称两种loss。

[Waterloo视觉实验室官方code目录](https://vision.cs.uwaterloo.ca/code/)明确归属实现者Meng Tang，并直接链接[meng-tang/rloss](https://github.com/meng-tang/rloss)。作者README说明原发表实验实现为Caffe，后来发布PyTorch版。本次同时读取两者，未用第三方复写充当官方实现。

固定源码commit：`1caa759e568db2c7209ab73e73ac039ea3d7101c`。6份源码及README在[源码快照](rloss_sources/)，哈希见[收据](rloss_sources/SOURCE_RECEIPT.json)。

## 它实际怎样利用灰度或颜色

直觉是“距离接近、颜色相像的像素应更倾向分为同类”，而不是“预测肿瘤的平均亮度要等于真值平均亮度”。代码实际用于二维RGB自然图像；单通道MRI和三维空间都需要另行改造，不能称现成BraTS实现。

1. 作者训练代码先把网络输入反归一化用于颜色相似度；分割分支传softmax概率，不传硬阈值mask。见[train_withdensecrfloss.py第125–130行](https://github.com/meng-tang/rloss/blob/1caa759e568db2c7209ab73e73ac039ea3d7101c/pytorch/pytorch-deeplab_v3_plus/train_withdensecrfloss.py#L125)。
2. 图像范围是0–255，颜色带宽sigma_rgb=15，空间带宽sigma_xy=100；示例权重2e-9，缩小倍数0.5。这些是该RGB示例参数，不是MRI通用值。见[作者PyTorch说明](https://github.com/meng-tang/rloss/blob/1caa759e568db2c7209ab73e73ac039ea3d7101c/pytorch/README.md#L25)。
3. C++按(x/sigma_xy,y/sigma_xy,R/sigma_rgb,G/sigma_rgb,B/sigma_rgb)构造特征，用高维双边过滤计算每类的AS。见[bilateralfilter.cpp](https://github.com/meng-tang/rloss/blob/1caa759e568db2c7209ab73e73ac039ea3d7101c/pytorch/wrapper/bilateralfilter/bilateralfilter.cpp#L3)。核的目标亲和度形式为A_ij≈exp(−||position_i−position_j||²/(2sigma_xy²)−||RGB_i−RGB_j||²/(2sigma_rgb²))；源码使用permutohedral lattice快速近似过滤，而非显式构造巨大A矩阵。
4. PyTorch先用ROI（有效区域掩膜）屏蔽padding，forward计算`−dot(S,AS)/N`；N为图像数，只按batch平均，没有对空间像素数逐像素求平均。见[DenseCRFLoss.py第21–40行](https://github.com/meng-tang/rloss/blob/1caa759e568db2c7209ab73e73ac039ea3d7101c/pytorch/pytorch-deeplab_v3_plus/DenseCRFLoss.py#L21)。因此它的小乘数不能与本项目逐病例均值MSE的权重1直接比较。
5. 输出缩小到0.5时，sigma_xy也乘0.5；sigma_rgb不随之缩放。图像及ROI使用默认插值，soft概率用双线性插值。见[同文件第58–64行](https://github.com/meng-tang/rloss/blob/1caa759e568db2c7209ab73e73ac039ea3d7101c/pytorch/pytorch-deeplab_v3_plus/DenseCRFLoss.py#L58)。

令M为ROI，S=MP，则代码计算的正则项是：

`L = −weight / N × sum_{images,classes} Sᵀ A S`

此处的负号不能当成损失反写；最小化负相似度是在奖励相似像素拥有一致类别分布。该项须与分割监督一起使用，单独使用不能保证有意义分割。

## 梯度不是被numpy转换切断

源码forward内确实转CPU numpy做过滤，但它是自定义autograd.Function，并实现了backward：`grad_segmentation = −2 * grad_output * AS / N`，随后乘ROI；图像、sigma和ROI不返回梯度。[DenseCRFLoss.py第42–47行](https://github.com/meng-tang/rloss/blob/1caa759e568db2c7209ab73e73ac039ea3d7101c/pytorch/pytorch-deeplab_v3_plus/DenseCRFLoss.py#L42)。这使loss梯度能回到softmax与网络，不能看到numpy就误报“梯度断了”。图像亲和度是固定输入，未要求对影像像素反传。

原Caffe也按batch平均、禁止向图像输入反传、向概率反传；Compute_DenseCRF同样计算负SAS，见[原始Caffe实现第101–149行](https://github.com/meng-tang/rloss/blob/1caa759e568db2c7209ab73e73ac039ea3d7101c/deeplab/src/caffe/layers/densecrf_loss_layer.cpp#L101)。

## 对当前灰度复现的意义

| 项目 | 作者DenseCRF | 当前灰度均值项 |
|---|---|---|
| 信息 | 成对像素空间距离和RGB差异 | 预测区域与真值区域两个均值 |
| 监督 | 正则项本身不需要整幅真值 | 需要真值区域均值 |
| 预测 | softmax概率，不硬化 | sigmoid软概率，不硬化 |
| 灰度/颜色尺度 | 反归一化0–255并绑定sigma_rgb | 标准化MRI均值，weight=1 |
| 归约 | 对所有类和像素的二次型求和，再除batch | 每病例区域均值平方差，再除batch |
| 空间 | 原官方实现二维RGB | 本地三维T1ce |
| 成本 | 训练增加双边过滤及CPU/GPU搬运 | 训练增加体素求和，无额外推理前向 |

真正可借鉴的是：把灰度尺度、soft概率、有效区域、归约、梯度路径与权重写成可核验实现，且空间结构可以进入约束。这不是师兄公式5的另一份“正确实现”，也不能由顶会自然图像提升保证BraTS提升。将其替换当前均值MSE会变成新方法，当前未实施或启动。
