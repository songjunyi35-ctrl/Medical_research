# T1ce-only与FLAIR-defined WT：一手文献边界

这份材料只解释WAW尾部，不提出或授权新方法。

1. BraTS2024 post-treatment官方任务把Surrounding Non-Enhancing FLAIR Hyperintensity（SNFH，周围非强化FLAIR高信号）定义为T2/FLAIR异常，包含放疗相关高信号、胶质增生、水肿和非强化肿瘤。因而本项目WT把SNFH纳入目标，而单T1对比增强并不直接提供其定义模态。[BraTS2024官方挑战论文](https://arxiv.org/abs/2405.18368)
2. 《Modality redundancy for MRI-based glioblastoma segmentation》的原始消融发现，T1CE单独可满足enhancing tumor/tumor core的最小输入，但whole tumor的最小输入是FLAIR（nnU-Net）或T1CE+FLAIR（SwinUNETR）；两模态组合在WT上接近全四模态。[原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC11442599/)
3. 《Brain tumour segmentation with incomplete imaging data》报告单T1CE的WT Dice仍约0.908，加入FLAIR后约0.943、接近四模态0.945。这说明T1CE-only不是逻辑上不可能，而是平均信息减少，某些纯FLAIR病例风险更高。[原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10144694/)

与本项目的连接：WAW only-label2比例较train高且四个共同零Dice属于only-label2，与“缺直接FLAIR信息”相容；但论文使用不同人群、全新训练、多类模型和协议，不能把其绝对Dice搬到本项目，也不能据此宣称WAW失败由单一模态因果决定。未来若目标允许多模态，必须重新定义输入、从零训练并使用新开发/外部证据，不能在已见WAW上直接加FLAIR调参。
