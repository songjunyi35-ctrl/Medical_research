# ROUND020：八预测的有符号距离形状融合

主来源Rohlfing/Maurer Shape-based averaging(2007) https://pubmed.ncbi.nlm.nih.gov/17283774/：用有符号Euclidean距离表示标签形状再融合；原论文不是当前同一CNN视图实验，本轮只借鉴表示，不称复现。

同一w32第171轮已存8镜像各在0.2/0.5/0.8阈值二值化；每组计算8张inside EDT－outside EDT，单位为原native1mm体素。候选有预测一致性权重lambda=-16/-8/0/4/8/16/32的均值，per-voxel trim6/trim4/median/25%/75%/min/max。42个场×15个全局距离level=-3至3=630配置。GT只在field返回后评分，不决定单例融合/阈值/point；point保持原mean8概率argmax，仍8次同权重部署前向、计算EDT额外CPU代价单列。

相同输入形状零level保持、重复完全一致、直算阈值与充分计数严格相同门通过。预测mask只在原worker/tmp：把它复制回主机被自动审批拒绝后，未执行/未重试传输，改原worker4CPU完成。与两卡相位及CPU6 DenseCRF并行，模型新增前向0、无训练。所有270同一全局规则，完全test-tuned；正负630与全部逐例矩阵保存。
