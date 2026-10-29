# Gate-only收益校准：CPU只读可行性审阅

本审阅没有实施、启动或占用GPU，没有读取正在覆盖的latest，没有对当前模型做新前向。现有full80保持原冻结继续。总控最新要求先清理磁盘，第二候选暂不实施。

**结论：有独立的机制隔离依据，尚无“已经证实基座漂移”的实证。** 可以把它预先登记为唯一第二候选，检验在固定e164基座与proposal时34参数校准是否有用；不能为了消除0%利用率，或仅因为头17轮分数下降而扩组。

e164正常与base原生108 Scan分别0.763266263、0.763265889，显式修正贡献约3.74e-7；因此保护这个已承担绝大部分成绩的基座，有事前结构依据。full80的gate监督虽然对收益目标停止梯度，但gate logits仍沿evidence/reference/backbone反传；base与ungated proposal也随联合训练变化。冻结这些模块可使同一增强输入的基座、proposal及收益目标保持固定，隔离门控校准和共同适应，属于与full80不同的优化假设。

头17轮训练内最好为新增12轮Scan0.760027，对e164差−0.003239；验证gate均值从e164的9.73e-5升至0.08145–0.09085。训练校准损失1.76244→0.69966，监督损失0.19097→0.17563，reference query总梯度范数中位0.004688、范围0.00412–0.01593，证实非gate模块也在共同更新。**这些日志没有逐轮base-only/mismatch Dice、基座输出漂移或校准损失专属backbone梯度分解，不能认定下降由基座漂移造成。** 校准损失下降和gate变大同样不能证明Dice收益。头17轮完整日志快照和输入SHA见本目录JSON。

## 一个候选的建议配方

- 固定原e164 SHA4bd6c35f…起点，仅gate_head.weight的33个系数和bias的1个系数训练；其余18,811,107参数冻结。project也冻结。所有冻结模块保持eval，backbone dropout关闭。Conv gate_head即使处于eval也仍可求梯度，不需要把整模型切回train。
- 保持原972/108、单T1ce、原生评分、同一增强和case顺序、global8、两rank、每轮972各一次/122更新；建议先保持微批2、每rank本地4、尾global4。新建仅含两个gate参数张量的AdamW。
- 单一前瞻LR2e-4，前10轮恒定，之后cosine到2e-5，weight decay1e-5、clip5；不做LR网格。该值为头部单独校准的工程候选，与历史RACE原训练率相同，是full80微调率的10倍。它与冻结eval模式一同构成候选差异，不能称只改变freeze mask的纯因果消融。
- 保持现有`.5DiceCE(final)+.5DiceCE(base)+.05contrast+.05gain_calibration`。冻结后的base和contrast是梯度常数，可继续计算/记录以便对照；仅final监督和gain项更新gate。收益仍是停止梯度的ungated-proposal局部BCE正负平衡，margin.002，中性处忽略；偶数更新增加原AP错配分支并按真实收益赋目标，不设gate floor/均值目标、不假定正常参照都可信。
- 建议80新增轮/9760更新，但只在本候选完整合成吞吐、数据和当前期限门后正式冻结。108选最大raw Scan，同分最早且e0可选；晋级沿用+0.003、seed20261029患者整簇CI下界正、Subject/Point不降。最终仍由总控统一限制最多两个270配置。

这只是候选配方，不是开训授权；详见PROPOSED_RECIPE.json。没有证据保证gate-only更好。BCE收益不是Dice收益，冻结基座也不保证加上修正后的Dice不降；若固定proposal不可用、34维线性门不可分或均衡目标与Dice不一致，该方案仍可失败。

## 实现与验收边界

现有runner不能直接改requires_grad后复用：它会调用model.train()重新打开冻结backbone的dropout，并读取correction/query等冻结参数的grad，导致错误。必须另建独立入口与实验目录，不修改现有80代码/冻结。冻结特征用torch.no_grad计算，gate_head在可求梯度区域运行；不要直接把inference_mode产生的张量送入需保存输入求weight梯度的head，除非先在该模式外clone为普通张量。

完整原生合成门须证明：trainable恰34；只有gate参数梯度/更新；其余全部参数逐字节SHA不变；固定输入的base logits与e164 eval逐位一致；正常/错配/无有效参照和尾4有限；五次真实合成更新后两rank全模型SHA一致、CPU严格模型/优化器重载通过、每卡<=38GiB。只训34参数不等于省掉SegResNet前向，每次不同增强仍需执行冻结网络；不能把已有测试或8镜像缓存冒充训练输入。

## 80轮的硬件可行性

full80头17轮去掉首轮后完整epoch中位172.12秒，80轮约3.825小时；旧准备阶段972+108及SHA约50分钟。gate-only省去基座反向传播，结构上应更省算力/显存，但本次审阅未跑CUDA，因此不提供伪造实测或保证。以full80实际耗时乃至既有约7.5小时保守预算作代理，再加约1小时独立缓存和导出余量，原则上可放进1154121尚余的初始24h资源；正式80仍须新合成吞吐核定，不预支条件96h。

若总控日后决定执行，须先独占重新分配1154121双卡并暂停A/B排队入口，核验SSH/CUDA/UUID/真实空闲和磁盘。该worker/tmp的972必须独立有界重建并核实际payload SHA/bytes/病例/几何，已有270缓存不能充作训练缓存；若复用108也须逐文件核验。完整best/latest优化器与恢复资料仅scratch，少量portable及文本必须重新分配总持久预算并维持>=20GiB。现有full80绝不暂停、换参数、缩预算或跨worker借卡。
