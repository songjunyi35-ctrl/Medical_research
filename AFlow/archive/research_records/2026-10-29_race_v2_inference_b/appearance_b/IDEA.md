# 七问提案：同图像外观似然校准（root已批准，所有常数患者评分前冻结）

1. **要解决什么**：固定075融合81.37%既存错误处在双专家均logit的正确侧凸包之外；单纯换权无法纠正。已选guided输入4项全部负，不能沿同一滤波网格无界加参。考虑从同图像前景/背景的外观密度添加有界连续unary，而非继续只在两输出间插值。该方向不能恢复单T1ce未包含的组织信息。
2. **一手依据/限制**：[Rother/Kolmogorov/Blake SIGGRAPH2004原文](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf)使用前景/背景外观密度与空间正则；其交互初始化和graphcut不是本提案的内容，也不是本任务收益证据。[Agn等MedIA2019原文](https://users.aalto.fi/~vanlek2/AgnMEDIA2019.pdf)使用生成式MRI外观模型并指出肿瘤与正常组织强度可重叠；其多模态、atlas、训练形状模型均不引入本轮。
3. **历史去重**：rg旧auto12h及旧validation scripts的GMM/GaussianMixture/appearance/likelihood无匹配。旧MRF/DenseCRF仅基于固定预测unary做空间消息；旧SDF/ROI/phase与温度/阈值/置信权重不拟合同图像类别外观。当前contra对比对侧局部强度差，cov用镜像扰动相关性，guided更改网络前图像；本提案不做空间图或新网络输入，只加入外观密度比。不是首创声明。
4. **已批准冻结的精确函数**：pR/pW为各8视图均logit的sigmoid。S=I!=0；FG种子S且pR>.95且pW>.95，BG种子S且pR<.01且pW<.01；FG<64或BG<1024一律回退既有cov，不强行开支路。每类以256bin图像强度hist拟合K=1或3的1维GMM；hist范围为本图像S内观测min/max（无GT）。均值初始化为该类hist的(k+.5)/K分位，权重1/K，方差全类方差下限.01；固定5次EM，方差下限.01、mix权下限1e-6并归一。拟合结束后在相同256bin中心计算归一化密度f/g，BC=sum sqrt(f*g)，trust=clamp(1-BC,0,1)。r(I)=clip(log p(I|FG)-log p(I|BG),-6,6)。score=既选covscore+λ*trust*r*S；λ=.25/.5。四项恰K{1,3}×λ{.25,.5}，除这些两项外所有常数冻结。score>0为mask，Point实际score.argmax。
5. **部署与资源**：读取原RACE8+w32base8，只接受image/logits，无GT/ID/逐例阈值；没有第三神经模型或第二SegResNet训练。GMM参数仅是确定性同图像后处理的中间统计，拟合不更新网络参数。新增神经前向0，部署2模型16次网络前向；GPU1估计2–5分钟108/4项，缓存掩膜只worker/tmp。本阶段GPU门和预算门通过后108；18:55硬截止。
6. **数值/选择门**：CPU先检查常量分布/同分布trust0、方差和mix正性、logsumexp finite、FG/BG交换符号、样本计数不足回退、几何动态和最终Point；root批准精确配方后GPU160/168纯合成门、再108患者固定五折。旧cov/fixed075/contra/guided/equal8为冻结参考，全部Scan/Subject/Point/患者区间/变差；新旧方法竞争，不能强迫新项入选。GT只scorer；oracle只诊断不入榜。
7. **何时失败/风险**：联合强自信错误会污染种子；神经输出和外观同来自这张图像，不能当独立证据或正确Bayesian后验。仅称启发式unary校准。若外观重叠或种子不足则回退，若新项低于旧参照不推荐；不按GT挑种子、病例、阈值或方法，不加新模态/模型，不自动270。


执行补充：采样不丢任何种子，使用完整原生crop<=5,591,040体素的256bin直方图，无随机子采样；核数/顺序不进入配方。GPU完整160/168吞吐不能在18:55前完成则不访问患者评分。该family后不再新增网格。
