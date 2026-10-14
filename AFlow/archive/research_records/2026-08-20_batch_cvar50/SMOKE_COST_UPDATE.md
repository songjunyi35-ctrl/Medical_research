# Batch-CVaR50 GPU smoke与成本更新

- 15项合成测试通过。
- 第一轮smoke缺GPU legacy parity审计；补审计后的v2发现FP32 per-draw与历史autocast DiceCE的loss差`3.35e-4`，按预注册停止。
- train-only诊断确认在同一autocast语义下，per-draw mean与legacy loss/gradient差均为0；修正输入精度后v3双臂全部重跑。
- v3 uniform legacy loss和raw-logit gradient差均精确0；两臂首batch pairing完全一致。
- peak allocated约`20.41/20.23GiB`，低于24GiB门。
- step中位时间约`0.353/0.353s`，32轮外推约`17.70/17.68min`。
- CVaR首batchstable top4索引正确，未选draw gradient为0，所有loss/gradient finite。
- 决策：不改top比例，允许双臂formal。
