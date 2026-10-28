# ROUND018：原生3D全连接DenseCRF

此前local_mrf只是6/26邻域候选，不冒充全连接CRF。本轮参考Krähenbühl/Koltun2011 https://arxiv.org/abs/1210.5644 与DeepMedic脑病灶 https://pubmed.ncbi.nlm.nih.gov/27865153/；使用pydensecrf2 1.1的通用DenseCRF(N,2)，XYZ+单T1ce强度4维双边特征，原native1mm去填充栅格，明确不是2D/RGB接口、无下采样/GT定位。MONAI1.5.2本环境未编译_C，未升级共享库；只在私有/tmp构建MIT/BSD C++实现，wheel/source记录见DENSE_CRF_VENDOR_PROVENANCE。无额外神经网络权重。

一元图由已缓存8视图当前一致性加权预测经当前温度4/偏置-1的概率变换生成，F64后F32、无GT，Point保持原mean8点。全局候选：无边项控制1个、仅Gaussian5/10轮2个；双边空间2/5/10mm×强度sigma0.25/1×权重1/3，各5/10轮共24个；Gaussian固定1.5mm/权重1。27×39阈值=1053，前8＋无边项控制后6375校准。零边项可与原解码因概率表示/浮点不同，不把这部分变化归因CRF。

小图无边恒等误差5.96e-8、重复逐位、完整160208168合成5/10轮10.44秒/RSS1.39GiB门通过。13图×最大10轮×270/6进程约1.7小时保守墙钟，实际crop更小；只有模型前向0的CPU后处理，不中断两卡相位。所有270同一全局参数，GT仅评分与全局选择，全部test-tuned；负结果完整保存。原生图像/标签/模型/core logits不变且不删。
