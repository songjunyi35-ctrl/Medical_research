# VM Dice paired GPU smoke成本更新

- 19项合成测试通过。
- 首轮gamma1 smoke在反传前因`torch.quantile`不能处理33,554,432个诊断元素停止；loss公式未改，smoke-only诊断改为exact min/max后两臂全部重跑。
- gamma0/gamma1首batch case、raw/augmented image和label hashes完全相同。
- 两臂peak allocated均`22,120,189,440` bytes，约`20.60GiB`，低于24GiB停止门。
- 中位step时间`0.3567/0.3575s`，32轮机械外推`17.88/17.92min`，低于45min门。
- gamma0/gamma1 smoke VM loss=`0.09323/0.92650`；gamma1总grad norm=`1.7859`，低于clip5且finite。
- modulation detached、正常logit路径gradient finite/nonzero、loss不消耗CUDA RNG、reload/deploy均通过。
- 决策：保持论文默认gamma1，不根据smoke调权；允许paired formal。
