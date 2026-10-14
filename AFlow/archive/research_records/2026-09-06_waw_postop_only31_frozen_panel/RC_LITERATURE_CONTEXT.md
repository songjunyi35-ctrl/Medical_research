# 早期术后切除腔Point失误：一手文献边界

本文件只解释已冻结WAW-postop31结果，不提出或授权新后处理。

- BraTS2024 post-treatment官方把resection cavity（RC，切除腔）新增为独立label4，并同时覆盖recent与chronic cavities；本项目binary WT明确排除4。因此RC不是普通背景，而是任务刻意区分的术后结构。[BraTS2024官方论文](https://arxiv.org/abs/2405.18368)
- 340名post-operative GBM的原始研究报告resection cavity、cerebrospinal fluid（脑脊液，CSF）与necrotic core之间存在混淆；加入surgery type与显式CSF segmentation能降低错标。该研究也指出腔壁可强化，最终判断依赖专家。[原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC9669429/)
- MU-Glioma-Post数据论文列出常见自动错误：RC内T1高信号血液产物被误作增强/非增强肿瘤，以及邻近脑室被误作RC。这与“最高响应落在腔体区域”相容。[数据论文](https://www.nature.com/articles/s41597-025-06011-7)

与本项目连接：WAW-postop31的Full/GCSAM Point miss约80%落GT RC，说明binary locator虽能得到约0.59 Dice，最高raw logit仍易被术腔相关信号吸引。但上述论文使用不同模型、标签和多模态输入，不能证明本项目因果。GT RC在部署不可用；任何自动RC head、CSF输入或手术信息都属于新模型/新输入，必须重新训练并在新证据上预注册，不能在已见31例上补后处理。
