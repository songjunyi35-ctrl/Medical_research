# 已审计原生3D DenseCRF `0019_dense_unary_control`

固定w32第171轮/T1ce、270/119，8个同权重视图的一元概率＋原生3D Gaussian/双边图，无GT输入或模型训练。

Scan **0.783491753**，Subject **0.770633068**，Point **258/270**。270个最高单图的独立重算与搜索矩阵逐例精确一致。额外CPU推理有成本，不能把网络8次视图等价当整算法零开销。

全局参数直接从复用270选择，test-tuned，非独立泛化。须与同图无边项控制比较，不把一元表示变化全部归因CRF。

[逐例](RAW_TEST270.json) · [配方](RECIPE.json) · [结果](RESULT.json)
