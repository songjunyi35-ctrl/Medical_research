# ROUND008：官方RankSEG BA/RMA

来源https://www.jmlr.org/papers/v24/22-0712.html 及官方https://github.com/rankseg/rankseg，RankSEG0.0.8 wheel只下载解压到本轮worker/tmp，源码SHA与BSD许可记录，绝不pip升级共享Torch/Triton。BA/RMA均为近似解码，不能称精确Poisson-binomial枚举。官方TRNA在10000体素CPU合成约2.47秒，BA约0.016秒；本轮先比较官方BA/RMA，可据预算另评TRNA。

概率配方从同270固定校准中选全局前2，加原五镜像/全八镜像，重合去重，最多4个。每个2solver×5温度0.75/1/1.5/2/3×4smooth0/10/100/1000=40配置。使用单通道multilabel、float64、pruning0、RMA关闭screening（官方明确其screening不承诺逐位相同）。温度全局固定，GT不参与解码或路由；Point保持相应概率argmax。每例全局同规则，全部结果test-tuned。

整脑5591040体素合成RMA连续两次mask完全一致约0.013秒，BA约0.230秒，峰值<1GiB；有真实第一例重复掩膜门。只复用core已保存logits，无新增模型前向/训练/outer/confirmation。正负全部参数及逐例矩阵SHA保留。
