# DMRFB paired GPU smoke成本更新

- 11项合成测试通过。
- 两臂adapter tensor初值一致，CUDA BF16下plain/control/candidate初始final logit差均为0。
- Step1仅projection weight/bias有finite nonzero梯度；step2三条branch、GroupNorm和projection全部参数均有finite nonzero梯度。
- 两臂首batch pairing完全一致。
- Peak allocated `21,818,996,736` bytes，约20.32GiB，低于24GiB门。
- 32轮机械外推约17.81/17.87min，低于45min门。
- 决策：保持dilation、双LR/WD和32轮不变，允许paired formal。
