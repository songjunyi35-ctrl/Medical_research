# 镜像前后答案一致训练：最终结果

状态：完成并关闭。平方分母soft-Dice一致性确实让模型更“镜像一致”，但两个全新患者隔离复制的准确率门失败；不进入任何旧validation/test/external cohort。

## 初学者版结论

**一句话idea：** 训练时让模型把MRI正着看和镜像看，镜像答案翻回来后必须一致，希望正着看一次和固定8镜像平均都更准。

**只改变什么：** Control与Candidate每步都做相同两次student forward；模型、数据、初值、sampler、增强、teacher、DiceCE加逐体素知识蒸馏、AdamW、168轮完全相同。唯一变化是平方分母sigmoid soft-Dice一致性梯度权重从`0.0`变成`0.1`。

**为什么不是旧KAS的重复：** KAS在epoch200后匹配深层feature投影并混合MSE/Dice/contrastive；本方法从随机初始化全程只约束同一final mask的可逆翻转等变。旧负结果没有提前否定本实现。

## 数据与访问顺序

- Replicate C/D各864 fit/108 inner、各49个inner subjects。
- C/D inner case和subject overlap均为0；还排除了已访问Flip8 A/B inner的93个union subjects。
- 四个模型全部固定168轮完成并冻结后，才写`INNER_PREACCESS`并构造C/D inner loader。
- 本实验访问了fresh C/D inner；没有访问fixed validation108、outer、confirmation、test270、UCSD75或其他外部cohort。

## 结果

### Replicate C（108 scans / 49 subjects）

| 推理 | Control Scan/Subject/Point | Candidate Scan/Subject/Point | Candidate−Control |
|---|---|---|---|
| identity 1× | `0.737180/0.731148/101` | `0.737496/0.732534/102` | `+0.000316/+0.001386/+1` |
| Flip8 8× | `0.742552/0.736345/100` | `0.740779/0.736987/102` | `-0.001773/+0.000643/+2` |

Flip8 Scan/Subject paired 95% CI=`[-0.006636,+0.003320]/[-0.005893,+0.007807]`；Scan点估计和两项CI安全门失败。Identity虽三项同向，但CI均跨0。

### Replicate D（108 scans / 49 subjects）

| 推理 | Control Scan/Subject/Point | Candidate Scan/Subject/Point | Candidate−Control |
|---|---|---|---|
| identity 1× | `0.737867/0.702368/104` | `0.737321/0.702453/103` | `-0.000546/+0.000085/-1` |
| Flip8 8× | `0.741552/0.707340/106` | `0.739488/0.703373/105` | `-0.002063/-0.003967/-1` |

Flip8 Scan/Subject paired 95% CI=`[-0.005778,+0.001629]/[-0.009182,+0.000867]`；Scan/Subject点估计与CI安全门失败。Identity Scan也下降。

### 跨复制

Candidate−Control平均：Flip8 Scan/Subject=`-0.001918/-0.001662`，identity Scan/Subject=`-0.000115/+0.000735`。四个跨复制门只通过identity Subject一个，整体`all_gates_passed=false`。

## 机制是否成立？

机制层面成立：最后5轮一致性loss的Candidate/Control比例，C=`0.6361`、D=`0.6480`，即候选减少约35–36%的镜像差异；两复制机制门都过。工程层面也全部通过：同初值、sampling weights、168轮sampler、增强RNG、reflection schedule、首batch与首existing/consistency loss均exact。

但科研直觉不成立：模型可以变得更一致，却把部分错误也一致地重复。C的8× Scan下降，D的1× Scan及8× Scan/Subject下降，因此不能把“减少变换方差”等同于“提高分割准确率”。这关闭当前`weight=0.1`、平方soft-Dice、单reflection循环的实现，不证明所有群等变网络或其他一致性设计永远无效。

## 实现失败边界

原预注册普通soft-Dice分母在预测完全相等时仍非0，会混入置信度锐化。该问题在任何训练/C/D inner访问前发现，透明冻结`AMENDMENT_01`改为平方分母，相等loss严格0。第一次V2 train-only尺度命令仅在最终向量转标量打印时报错，没有产生可用指标；修正后在同64个train病例冻结尺度与weight。没有隐藏或用inner修公式。

## 代价和下一决定

- 四条训练合计约20.873 GPU-hours，两波双A100墙钟约10.484h；每卡峰值42.006GB。
- inner评测共3,456次locator forward；workspace约110MB，归档写结果前56KB；无新cache、无删除。
- 不调weight、公式、reflection schedule、epoch、seed或阈值，不加第三复制，不进任何旧cohort。
- 用户已授权GPT‑6继续30小时长周期科研；下一方法必须重新完成七问、预注册、一手文献、fresh证据和单因素门槛。

## 证据

- [七问](IDEA.md)
- [原预注册](PREREG.md)
- [训练前修订](AMENDMENT_01.md)
- [实现凭证](IMPLEMENTATION_RECEIPT.json)
- [formal launch](FORMAL_LAUNCH_RECEIPT.json)
- [inner preaccess](INNER_PREACCESS.json)（SHA-256 `d24c5cb3…1662d`）
- [inner started](INNER_STARTED.json)（SHA-256 `4fd19f98…6152`）
- [raw inner](RAW_STAGE_A_INNER.json)（SHA-256 `82913854…eef67`）
- [机器结果](RESULT.json)
