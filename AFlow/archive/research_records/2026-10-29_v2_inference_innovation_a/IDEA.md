# RACE同模型八视图：局部相关分歧融合

1. 问题：当前108最强单模型解码是equal8 logit均值0.5阈值，Scan0.768011881。旧全局/逐点加权没有检查空间局部的跨视图共同分歧。目标是在不加网络前向的条件下检验局部依赖建模，实测为准，不保证提分或超过0.8。
2. 依据：Wang等[Joint Label Fusion原论文](https://ieeexplore.ieee.org/abstract/document/6226425)对相关atlas错误显式建模，说明独立可靠度权重可能遗漏共同错误。这里改用同一RACE八镜像的留一多数票分歧代理，不是有真实错误标签的原JLF复现，也不主张首创。各视图高度相关，代理可能共同犯错。
3. 方法：m_i=1[z_i>0]，e_i=1[m_i≠majority(other7)]。9/17mm有效box平均估计M_ij=avg(e_i e_j)，在4voxel格点计算(M+ridge I)^−1·1，截负归一后按原物理格点三线性展开。仅在投票分歧且|equal8 logit|≤2处，以blend×clip(weighted−equal8,−2,2)更新。阈值固定0；Point主指标使用实际refined score argmax，equal8 point另列辅助，不择GT更优点。
4. 唯一研究变化和去重：full joint与仅对角可靠度为机制控制；依赖模式2×box2×ridge{0.05,0.2}×blend{0.5,1}=16精确项，其他常数全前瞻固定。旧case-consistency为整病例标量，旧voxel为独立p(1−p)权重，旧joint只是二者相乘，EM为全局混淆率，MRF平滑融合结果，SDF为全局距离聚合；均没有局部8×8依赖求解。留一均值残差只是8/7倍普通残差，故不以该代数重命名冒充新方法。
5. 数据与统计：固定RACE164和已有108/50八镜像只读logits，网络前向0。原单T1ce/native1mm/原始裁剪全GT，脑支撑外阳性仍计FN。部署deploy(logits,image,recipe)不接GT或ID。统一既有患者5折与10000患者整簇bootstrap seed20261029，以equal8为最强参照并纳入选择池；先四折选全局配置，再评价留出折。选择后区间描述性，继承test-tuned历史，不称独立确认。
6. 执行门和预算：仅1154121 GPU0，固定队列锁。9/17mm box采用可核对的三轴可分离平均，统计一次复用16解码。CPU/GPU数值门含box与直接3D参照、alpha0/unanimous恒等、非负权重归一、视图置换一致；纯合成native160×208×160测完整16项时延和显存后再冻结执行。估计108最多45分钟、reserved≤16GiB；新增持久≤64MiB、项目空闲≥20GiB，数组/掩膜只worker/tmp。记录CUDA event算法区间、墙钟、真实峰值，绝不循环无意义负载。
7. 成败与交接：完整保留16项逐例TP/FP/FN、Point_refined/Point_base、所有变差病例。共同错误无法从自一致性识别，局部估计和逆矩阵也可能放大噪声；若不胜equal8记录负结果，不扩网格。提供纯部署函数；未来新训练best只迁移本轮选定一个配方，不再次16搜索。新增270评分始终0，最终270须总控更新最终冻结政策。训练节点、PID和全部旧源码/32配置冻结不动。候选2 gate-only训练已按用户范围澄清取消，未启动。
