# 七问：双专家冲突处的对侧图像证据

1. **问题**：固定0.75融合与配对TTA协方差只观察预测，不能利用原图上病灶样局部非对称。研究单T1ce局部正向强度差是否能在冲突体素小幅改变专家权重，不称临床实证强化。
2. **依据**：[Bianchi等ISBI2013作者原文](https://montillolab.org/publications/Montillo-ISBI-2013-BrainTumorSegmentationWithSymmetricTextureDrivenDecisionForests.pdf)用对侧区域强度/纹理差与训练森林；区域平均减轻局部对应误差。[Ghanavati等ISBI2012原文](https://iacl.ece.jhu.edu/proceedings/isbi2012/pdfs/0000574.pdf)提示正常脑也不完全对称、单独强度特征不足。[Raina等2019原文](https://arxiv.org/abs/1907.08196)强调对侧对应配准；我们的简单平面只是代理，不能视为精确同源组织或首创证明。
3. **独立机制/历史去重**：旧790万为全局/置信度/ROI/相位搜索；local_mrf_v1为相邻Potts双边信息传播，既往最优退化为unary；旧SDF/形状不看对侧原图。当前不传播标签、不调温度或阈值，不微调协方差参数；区别RACE训练特征参照及A新输入矩归一化。旧失败全部保留。
4. **算法冻结**：image-only fit_plane在原生去填充crop重新拟合LR平面。整数反射x'=2c-x，image与非零支撑同变换，crop矩阵与canvas平移共轭CPU门明确验证。共同有效支撑内k=5/11局部加权均值及方差；Δ=μ(I)−μ(Iref)，s=sqrt((varI+varRef)/2+0.25²)，固定中心0尺度1。e=tanh(abs(Δ)/s)或tanh(Δ/s)。pR/pW=sigmoid(8视图均logit)；o=4relu(-(pR−.5)(pW−.5))；d=tanh((zR−zW)/4)。h=relu(1−max(pRref,pWref)/.2)，乘共同有效支撑与planeDice>=.8；缺失或疑似病灶参照回退。α=clamp(.75+β o d e h,.5,.95)，β=.1/.2。score=αzR+(1−α)zW，mask=score>0，Point=score.argmax。严格2×2×2=8，不看108后补阈值。
5. **部署/资源**：仅image+两专家各8同镜像缓存，16前向/例成本，无新NN、无第三模型、无GT/病例ID/查表/逐例阈值。GPU1独占锁，训练节点不动。缓存只读，掩膜worker/tmp，ROOT文本<=64MiB且free>=20GiB。108先选，当前不270。
6. **判定/风险**：同时报最强单模equal8、旧.75及cov冻结参考，Scan/Subject/Point、10000患者配对区间与全部变差病例。8新+2冻结参考患者五折选参。锚点/两旧参照已用全108，五折仅条件稳定性，不称fresh OOF或独立确认。边界/病灶/有效脑/冲突区域报告trust、o、Δα与回退，低全图均值本身不判失败。正常非对称、双侧病灶、坏配准、非强化肿瘤可造成错误，小幅界和健康参照门不保证获益。
7. **停止/失败**：CPU几何/符号/交换方向/对称回退与GPU160/168门通过才108；失败保留记录、无自动重NN。仅8配置，单一best_new供未来固定迁移，旧参照可胜出。无270自动访问、改口径或排除病例。
