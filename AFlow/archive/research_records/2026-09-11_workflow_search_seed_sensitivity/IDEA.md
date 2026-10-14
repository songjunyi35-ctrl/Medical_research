# 搜索随机性敏感度：主实验后的限定补充分析

Status: post-primary-result preregistered sensitivity, not independent validation.

## 1. 现在的问题是什么？

主工作流协议两模型均通过：2/3次前向，实测3.961/2.652倍加速。但16次查询的随机搜索在seed11点门失败，UCT在seed12点门失败；每折只用了一个搜索随机种子，不能据此概括算法优劣。

## 2. 一句话直觉

抽签只抽一次可能碰巧幸运或倒霉；固定多抽二十次看波动，而不是挑中最好的一签。

## 3. 为什么现在选它？

原始[UCT](https://sites.ualberta.ca/~szepesva/papers/ecml06.pdf)含随机探索，本项目冻结实现的实际14–15次树模拟非常少。随机增强策略搜索也应有随机基线。当前穷举与贪心在实现上确定，不需要虚构20个独立结果。该分析针对算法随机性，不复活失败配方，不证明新患者泛化。

## 4. 具体怎么做？

只读主实验128子集逐病例评分→固定原患者三组→random/UCT分别20次固定种子重放，每折16次唯一查询、8候选池、同selector规则→固定后读取同折assessment行→形成相同180/78 OOF结果。沿用主协议的患者簇区间和六项门。保留每次全部选择轨迹和OOF行，计算通过频次、Scan/Subject差值、Point、成本及search reward regret分布。

## 5. 只改变什么，哪些不变？

唯一变化为搜索随机种子。grid r=0..19；每折seed=2026091150 + 1000*r + 100*model_ordinal + fold。两个模型、128子集、患者三组、16查询、8候选、reward、selector与bootstrap种子2026091141+model_ordinal均沿用，不调整。原主实验exhaustive、greedy和原随机种子结果不替换；不选新final workflow。

输入主实验RESULT、ANALYSIS、RAW_DEVELOPMENT及两个RAW_SEED JSON全绑定SHA，读取内容前先写PREACCESS。只读存量统计，不加载checkpoint、影像、标签数组或22.5GiB原子。

## 6. 成功、失败与停止

这不是方法晋级，不改变主实验任何结论。描述指标：每模型和双模型联合20次门通过频次，各controller成本/Point/相对full8 Dice差的min/median/max，search regret。二十次共享完全相同患者，不给“20次外部复制”或把通过频次当临床置信概率。没有算法优越性显著检验，也不从20次选最优种子部署。任何来源SHA错误/主算法变化停止；全部20次完成后关闭敏感度分析，不扩种子网格救故事。

## 7. 代价

0训练、0GPU、0新模型前向、0新影像标签访问、0test/confirmation/外部。预计CPU数分钟，新增JSON预计<20MiB，无大型cache。主实验后的分析必须明确posthoc动机，不能冒充在主结果前冻结的消融。

## 8. 最终结果

待20次完整重放。结果独立归档，不覆盖主实验。
