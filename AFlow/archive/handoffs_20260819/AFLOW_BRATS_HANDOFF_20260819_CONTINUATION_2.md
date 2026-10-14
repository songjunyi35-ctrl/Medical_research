# AFlow BraTS continuation handoff 2 — 2026-08-19

本文件承接`AFLOW_BRATS_HANDOFF_20260819_CONTINUATION.md`，记录160³负结果之后的新科研。

## 29. true-128³ late-cosine EMA：能改善同seed raw，但未超过历史最佳

全756例、seed`20260833`、200轮true-128³训练完成。EMA在epoch100后初始化，epoch101起每步
decay`0.999`，更新数严格为9,400；raw/EMA epoch168/200均保存且reload差为0。同轨迹上EMA168
相对raw168的scan/subject/point增益为`+0.012271/+0.011547/+0.009259`，subject CI
`[+0.005038,+0.020030]`；EMA200增益仅`+0.000608/+0.000689/0`。

但在同一固定validation108/50 subjects上，历史最佳true-128³ epoch168为
`0.731601/0.722121/0.925926`，EMA168仅`0.718825/0.710218/0.944444`，scan/subject差
`-0.012776/-0.011903`且CI上界仍为负；EMA200更低。因此EMA单模型不进入test。归档见
`archive/research_records/2026-08-19_plain_ema/RESULT.md`。

## 30. 历史最佳 + EMA168固定logit ensemble：新internal test高点

观察到EMA168虽Dice较弱但point hit更高、误差互补后，在test前冻结线性logit混合：
`(1-w)*historical + w*EMA168`，候选权重仅`0.25/0.5/0.75`，固定validation按scan优先、subject
次级选择。最终选择`w=0.25`。validation相对历史模型提升scan`+0.002570`、subject
`+0.002369`、point`+0.018519`，paired CI下界均大于0，全部gate通过。

固定候选在复用internal test270结果：

| metric | historical epoch168 | w=0.25 ensemble | 差值 | paired 95% CI |
|---|---:|---:|---:|---:|
| scan Dice | `0.741203949` | **`0.745054064`** | **`+0.003850115`** | `[+0.003171,+0.004565]` |
| subject Dice | `0.723179486` | **`0.727344609`** | **`+0.004165123`** | `[+0.003141,+0.005269]` |
| point hit | `0.929630` | **`0.955556`** | **`+0.025926`** | — |

222/270 cases提高、46下降、2持平；100/119 subjects提高。point transition为249 hit→hit、
2 hit→miss、9 miss→hit、10 miss→miss。该结果是当前项目在复用internal test上的新高点，
但不是pristine blind test、外部确认或公开SOTA。代价是locator推理约2倍计算；无额外训练、TTA、
后处理或SAM。test不会回流调权重、checkpoint或阈值。完整归档见
`archive/research_records/2026-08-19_plain_logit_ensemble/RESULT.md`，test report SHA-256为
`b49cf72a7d749afd28bc3009d0f7ad27b63ce5642424cc8eeb183e4eec7dbc7c`。

当前推荐benchmark方法为固定w=0.25双locator ensemble；若下一步追求效率，应在新validation协议下
研究把ensemble蒸馏回单个4.70M locator，而不是继续增加分辨率或test上调权。
