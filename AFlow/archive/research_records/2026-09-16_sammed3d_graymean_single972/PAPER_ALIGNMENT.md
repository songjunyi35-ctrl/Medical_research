# 师兄原文参数核对与明确差异

核对来源：本目录MSCL_SOURCE.pdf，第3.1节；文字提取MSCL_SOURCE.txt第256—261行。原文明确：two Tesla v100 GPUs；batch size of 1；learning rate of 8e-4；weight decay of 0.1；for 200 epochs。

| 设置 | 原文 | 本轮 | 结论 |
|---|---|---|---|
| 训练轮数 | 200epochs | 总200完整轮 | 数值一致；本轮仅灰度单组，不是完整多阶段课程学习 |
| GPU | 2×Tesla V100 | 先单A100，用户授权后2×A10040GB | 两卡同一实验，不是两组；硬件不同 |
| batch | 1，未交代全局/每卡 | 前阶段全局1，随后每卡4/全局8 | 用户明确授权差异，不能称严格对齐 |
| 学习率 | 单一8e-4 | encoder8e-4，prompt/mask head8e-5 | head倍率沿本地官方入口，与原文单一lr有差异 |
| weight decay | 0.1 | 0.1 | 一致 |
| 输入形状 | 128×128×128 | 128³ | 形状一致；1.5mm/真值中心裁剪为本地具体约定 |
| 输入预处理 | normalization、intensity correction、skull stripping | 既有源影像+正值标准化，未额外做N4/直方图校准 | 不能凭名称声称全处理链一致 |
| 灰度项 | 第2.3节公式5，首次提示区域灰度均值误差 | 首次提示软概率灰度均值MSE，权重1 | 对应公式；权重/软硬mask/归约等细节本地补全 |
| 数据划分 | 未提供可逐病例核对的同一划分 | 用户固定972/108/270 | 固定本项目划分，不冒称作者相同病例名单 |

双卡数据并行按进程分别提供输入、同步梯度；每卡4例对应全局8例。实现依据：PyTorch官方DistributedDataParallel文档 https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html 。每轮972例无重复填充/无丢弃，末批全局4例；针对性测试核对与原DataLoader全局排列相同。

因此：200轮确实来自原文，不能再笼统宣称“全部参数已经对齐”。本次目标是固定972训练集上的SAM-Med3D灰度单组、用户授权双卡更大batch，报告必须列出这些差异。
