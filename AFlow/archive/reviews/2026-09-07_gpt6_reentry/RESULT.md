# GPT-6接回：当前证据与恢复模型开发的决策

状态更新：用户已明确回复允许修正后的864扫描/313与78患者方案，角色已冻结并开始配对训练。[授权](../../research_records/2026-09-07_reused_development_roles/AUTHORIZATION.md)、[split](../../research_records/2026-09-07_reused_development_roles/SPLIT.json)、[实验](../../research_records/2026-09-07_structured_background_aux/IDEA.md)。以下保留批准前的审计和方案背景；其中待批准/blocked描述属于历史状态，不代表当前科研停止。

## 已验证事实

读取Fable的SESSION、正式预注册、评测代码和raw后，独立从保存的intersection/predicted/target计数重算56 scans/41 subjects的三组Dice均值、Point及绝对/配对bootstrap，与raw一致。raw及source/cache/model-preaccess收据SHA与原终态验证一致。[新审计](AUDIT.json)。本次没有重跑模型、读取影像数组或新增测试性能。

Fable这轮没有新模型训练。它新增UCSD原挑战test公开份额的患者评测：Full identity/Flip8/GCSAM Scan Dice约0.730002/0.732973/0.717689。说明某些既有checkpoint的效果依队列可逆，不能等同于新算法提升。已有scan bootstrap独立重采样扫描；同患者重复扫描的依赖由单独的Subject统计处理，scan CI本身不能被称为患者聚类CI。

逐个复核历史排除manifest的SHA和患者集合后，972个训练扫描/444人中，在“永不复用曾揭示inner患者”的旧规则下确实只余02633和02870两人/3 scans。历史独立性无法靠更换随机种子或重新分折恢复。

## 可审阅的角色调整提案（尚未执行）

1. 只使用原972扫描/444人的train manifest，SHA=`6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`。本提案不把fixed validation108、outer、confirmation、internal test270或外部队列加入开发池。
2. 进一步metadata审计发现原972池包含旧confirmation全部53人/108扫描。先排除这些患者，实际可提议复用的开发池为864扫描/391人；批准后用固定seed一次性按患者划分313人fit、78人development validation，同患者所有扫描同组，实际scan数生成后冻结。此前355/89提案由[角色范围修正](ROLE_SCOPE_CORRECTION.md)取代；不静默吸收confirmation。
3. 允许在这个明确标记reused-development的固定split上做有限、逐个预注册的单因素方法开发。每项候选先声明网格/轮次/读取频率/停止门，记录累计开发访问；不再把每次看同一验证集写成fresh。
4. 每个新方法用matched control；需要教师时教师只能在当前fit患者上训练，既有全972 checkpoint不能当隔离教师或暖启动。先以标准原配方从随机初始化建立对照，再验证候选。训练随机种子的复核检验优化稳定性，不伪装新患者外部确认。
5. 首个候选的具体机制、论文、预算和成功门另行完成七问/IDEA/PREREG；本文件不预先批准任何loss、架构、crop、阈值或外部结果驱动的调整。允许研究不等于保证提升。
6. 开发过门只表明值得后续验证，不自动授权对已封存测试/外部集重新评测。若未来用户要求在旧test报告最终模型，必须另行明确reused/test-tuned范围；独立泛化仍需要新数据。
7. 代价：本提案本身0训练/0GPU/0cache。批准后的训练使用两张A100、现有128³训练cache；每个方法的GPU-hours/显存/存储预算须在正式运行前冻结。充足额度不会改变证据身份。

## goal状态

接手时get_goal为paused，累计83,770秒（约23.27小时），原目标30小时。随后本轮续接实查已为active；由产品恢复，未调用不存在的resume接口。界面恢复goal与科研数据角色授权是两项独立状态。角色授权仍待用户答复。

下一步：用户批准角色调整后冻结开发split并开展真正的单因素模型开发；若不批准，继续原范围的新来源/只读审计。当前未启动训练或重新划分病例。

本次再次接手及随后两次自动续接中，同一数据角色授权始终未收到；期间已完成保存统计复算、manifest边界修正和文献来源审查。当前无可接续训练任务、无获权的新外部人工队列。已达到连续三个回合的阻塞审计条件，goal应标记blocked等待用户批准角色调整或提供新的合法数据；目标未完成，不得标记complete。已准备提案限定864 scans/391人、313/78患者划分，不动旧confirmation/test/外部角色。

等待期间已完成[背景细分监督与TextBraTS来源审查](LITERATURE_FIT.md)：前者有一手SegResNet先例但尚无本项目效果证据；后者公开版本源自BraTS2020，期刊扩展未获新患者清单，不能冒充独立外部来源。
