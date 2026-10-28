# ROUND005：既有Dropout层的随机推理

Gal与Ghahramani的MC-dropout论文https://arxiv.org/abs/1506.02142 给出训练时Dropout条件下随机推理的依据；这里仅把原模型已有Dropout/Dropout3d层启用，所有权重固定且无梯度，归一化模块eval。候选p=0.05/0.1/0.2，16次=八镜像×两种随机掩码，batch8/尾6，固定种子。改变p只是敏感性候选，不宣称所有p都有原论文贝叶斯解释。

每组与固定五镜像概率的全局融合及校准沿ROUND004；全部270直接test-tuned，负结果保留。合成整batch与几何/有限值/显存门先执行。16次不是8次；成本记录需以views长度计，后续组融合行应从其原render任务取得精确前向数，不能照抄普通八镜像13次成本。
