# 2026-09-16 worker 接入与主线确认

用户已明确模型就是 SAM-Med3D，并授权自行查阅论文和公开灰度方法。已通过 worker 1097281 登录，项目共享目录可读写；nvidia-smi 和项目虚拟环境 PyTorch 均识别两张 A100-SXM4-40GB，探测时无 GPU 作业。当前约22 GiB空闲。

## 可复用连接方式

本会话 exec_command 在项目 cwd 下仍可能因 sandbox launcher 缺失而失败；已成功的方式是从 /tmp 使用 tty=true，运行 `mlx worker login 1097281 -- bash`，再进入项目。CLI 自带帮助确认支持 worker ID，无须用户提供 SSH 地址。Python 环境为项目旁的 sammed3d-venv，PyTorch 2.7.1+cu126。

## 历史清理与本次动作

现有 CLEANUP_RECEIPT.json / RESULT.json 记载删除328项、17.18 GiB，18份保留权重校验通过；本次连接只读取已有收据，没有再次删除或复算全部权重。原报告保持。GPU接通不代表训练已启动。

## 文献核对进展

- [SAM-Med3D 原论文](https://arxiv.org/html/2310.15161v3)：三维图像编码器、提示编码器、分割解码器；需固定交互点击和裁剪协议。
- [官方训练源码](https://github.com/uni-medical/SAM-Med3D/blob/main/train.py)：含正灰度体素统计的 ZNormalization；这是输入标准化，不能直接称师兄的灰度阈值损失。
- [Nyul 原始灰度标准化论文](https://doi.org/10.1002/(SICI)1522-2594(199912)42:6%3C1072::AID-MRM11%3E3.0.CO;2-M)：不同扫描的强度尺度标准化；目前仅为文献线索，未被选作正式训练方案。
- 历史 T1CE_MAINLINE 记录灰度项尚需核对 soft/hard mask、权重、归一化、空预测、归约维度及交互步。当前公开搜索未找到可确认的 MSCL 原文；不把搜索不到写成论文不存在，也不把0.731当可复现目标。

下一步继续追溯灰度方法及本地训练入口，按七问冻结可审计的方案，完成训练侧合成测试和GPU短试跑，再正式训练真正的SAM-Med3D。若采用文献替代方案，须标明其身份，不能冒称师兄原实现。

本次没有新训练、像素读取、检查点反序列化或验证/外层/确认/测试评估。无新Dice或置信区间；排行榜数值不变。访问证据见 WORKER_CONNECTION.json。
