# 自动模型的精度约束镜像子集搜索

Status: preregistered before implementation and formal development access.

## 1. 现在的问题是什么？

同两个fit684原生模型的无标签测速显示四视图相对八视图快1.993/1.994倍，但尚无删视图后的精度证据。旧GT中心修正器的搜索未胜随机/贪心，旧自适应路由未胜等成本随机，不能默认迁移为成功算法。

## 2. 一句话直觉

像删掉重复检查：保留提供不同信息的视角，而不是只少做几遍。空间足够小时全查一遍，能知道复杂搜索到底有没有帮忙。

## 3. 为什么选择这个idea？

[Greedy Policy Search](https://arxiv.org/abs/2002.09103)支持把测试时增强视图作为可学习策略研究，但它不是本项目三维术后单通道WT的精度保证；此处greedy是明确冻结的局部best-first子集控制，不冒充该论文精确复现。
[Kocsis/Szepesvari UCT](https://sites.ualberta.ca/~szepesva/papers/ecml06.pdf)提供探索/利用树搜索依据，不证明它在128个小空间优于穷举或随机。Monte Carlo Tree Search（MCTS，蒙特卡洛树搜索）中的UCT指把上置信界用于树上选择。本轮主方法是精度约束的有限工作流选择，不将树搜索本身包装为创新。
[官方镜像实现](https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/inference/predict_from_raw_data.py)与本项目历史有序float32融合固定数值语义。

旧负结果针对旧修正器/学习式路由，未否定当前自动模型固定子集剪枝；本轮不复活旧subject-balanced Flip8失败门，不改训练配方。

## 4. 具体怎么做？

两冻结native control模型分别处理同一复用dev180/78：原T1ce image-only crop128→八个固定镜像预测→各自逆变换为原坐标float32原子→仅对选定子集按原顺序等权平均raw logits→阈值0输出WT掩膜、全局argmax输出点。无教师、损失、训练、SAM（Segment Anything Model，提示式分割模型）或后处理。

镜像顺序：0=[]，1=[-1]，2=[-2]，3=[-2,-1]，4=[-3]，5=[-3,-1]，6=[-3,-2]，7=[-3,-2,-1]。候选ID k=0..127，包含视图0，另按k的第0..6位选择视图1..7；按视图编号有序相加。identity=k0，full8=k127。128种全计算只用于可审计评价缓存，不能称搜索过程省了GPU。

开发78患者按SHA256("2026091131:"+subject_id)排序，连续切三组，每组26人。fold f用组f评估、(f+1)%3搜索、(f+2)%3选择；所有扫描随患者走。模型训练未含dev，无教师；但这些患者之前已多次研究，不能称pristine或新的独立outer。本轮称nested reused-development assessment；旧protected outer/confirmation/test完全不访问。折内数据接口只给各阶段自己的行，保存调用/选择收据，选择不得读取评估组。

主控制器exhaustive评价全部128子集；random、greedy、UCT三控制器每折各16个唯一search查询（包括强制identity/full8两个参照）。random不放回抽14；greedy从已查候选中按reward优先扩展7位单bit邻居，邻居按编号，直到16；UCT在7层二叉包含/不包含树上，用均值reward+sqrt(2 log(parent visits)/child visits)，未访问分支先均匀选，叶子重复可用缓存，最多512树模拟或16唯一查询，两强制参照已入缓存但不虚构树访问。若不足16，报告预算未用满，不随机补齐伪装UCT。种子=2026091132 + 模型序号*100 + fold；不选择搜索随机seed。

search reward=患者等权Dice−0.001*视图数。每控制器只把search reward前6个非参照+identity/full8（共8）给selector，不按assessment改名单。selector可行条件：相对full8，Scan与Subject点估计均≥−0.003，Point≥max(identity,full8)−1；可行者取前向最少，再按Subject降序、ID升序。无可行者退full8，明确标fallback，不假装安全门通过。full8总在候选池。

各折固定选择后才拿本折assessment病例组成180行out-of-fold（OOF，折外预测）报告。主判定仅exhaustive，不能看结果把获胜次控制器换成主方法；次控制器全部报告成本、reward regret对全空间最大search reward、与主方法的配对描述。分别报告identity/full8及预定opposite2(k64)、single-axis4(k11)全dev固定参考，不据此择模型。均是同78患者，不合并两seed为156人。

## 5. 唯一变化与固定项

只改变执行视图子集/预定控制器，训练因素全部冻结。native control固定seed2026090711/12、epoch168，fit684/313，dev180/78；数据/权重SHA继承前阶段preaccess并重新计算。fit SHA75504df75fa734723945a9f07d76696fdcca89b8c29563dc0ad02a2560718d61；dev SHA53885b7e300f915d81eecb115b82123e25330b7da422d3ff0b889b0489a2cdac；index SHA7373cb306a01ec1ec311a584a0887466875（以下述机器收据完整原值为准，不用本句缩写作校验）。权重与cache绑定全部由既有structured_background_protocol及工程preaccess中的完整SHA逐文件核验。

batch8、bf16前向、float32融合、原始WT nearest-exact128评分，阈值0、点argmax不变；既有单次native逐病例结果要求完全相同，不相同即失败。fit smoke每模型前8例，8原子+历史8视图+在线预定1/2/4/8视图：每seed248，两seed496前向。正式原子收集2×180×8=2880模型病例前向；搜索/重算融合不增加模型前向，另记融合量。

## 6. 成功、失败与停止

Primary：exhaustive OOF对full8精度—成本。两seed各自要求：平均前向≤4；Scan/Subject点估计差≥−0.003；两患者簇配对95%区间下界≥−0.005；Point≥max(identity,full8)−1。10,000次患者簇bootstrap，seed2026091141+模型序号；不作多重探索校正、不声称正式统计非劣。OOF区间条件于已选择策略，未覆盖全流程选择不确定性。

通过只允许冻结六个fold workflow做fit-only真实在线复核与测速，不能部署或自动测test。在线相对原子需逐体素exact，selected workflow相对full8同病例中位speedup≥1.5在两seed都成立（平均fold策略成本；不按病例难度路由），含H2D/融合/mask-point D2H，不含预处理与模型加载。若精度失败，仍可无标签测量被选workflow作为负结果成本说明，不改变晋级结论。

任一主门失败关闭此固定协议，不调阈值/候选空间/seed/精度门救结果；次控制器偶然通过不能改主判定。MCTS未胜控制器如实记负，不影响“有限自动剪枝”本身的受限证据。正式开发/分析异常保留partial与访问边界；只修纯实现错误，原记录不覆盖。无test/外部/confirmation/旧outer访问准入。

## 7. 成本与风险

0训练。两seed原子float32共22.5GiB写/tmp，score JSON预计<30MiB；启动前/tmp现5.2TiB可用，root4.7GiB，不在root创建原子。无静默删除，缓存有SHA可重建但保留。正式开发模型前向2880，子集-病例融合46080；推理期安全上限2小时。峰值显存预计<12GiB，以实际为准。最终部署每病例1..8次，不在部署预计算全8再剪枝。

风险：反复开发偏差，搜索/selector小样本、纵向患者相关、Point与Dice冲突、控制器小预算难分优劣。穷举提供小空间参照，不将有限空间最佳或探索性OOF叫外部最佳。最终确认需要另冻合法协议，不以不停工作为由突破数据角色。

## 8. 最终结果

待执行；所有结果、未过门与实际开销均归档。
