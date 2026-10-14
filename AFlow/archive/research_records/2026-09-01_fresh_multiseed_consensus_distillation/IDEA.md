# Experiment idea: fresh三seed共识蒸馏回单模型

Status: planned and frozen before split generation, implementation, GPU model construction, training, or new evaluation access.

## 链路审视与启发

当前权威链路是：`raw NIfTI → subject级manifest/角色冻结 → N4 + image-only brain crop + true-128³ cache → random-init Full972 voxel-KD → train-only inner选epoch → fresh-seed full retrain → reused fixed validation108 gate → 仅明确授权才作reused test270报告`。

1. **更多病例从第一轮参与仍是最大可复现机制。** Full972相对same-seed/same-compute Full756 validation Scan/Subject为`+0.010943/+0.007982`，Scan CI整体为正。限制证据是Full972相对KD200 Subject CI下界`-0.004484`未过门，且test已复用。缺少独立新cohort确认。
2. **容量和全图分辨率不是当前主瓶颈。** Width24参数从4.70M增到10.57M，fresh-seed final反而`-0.002872/-0.003525/-1 Point`；true-160³约1.95×体素计算，test Scan从`0.741204`降到`0.737060`且Point下降。限制是这些只否定当前扩容实现；但下一步不应再盲目变大。
3. **三seed输出共识是目前唯一同时提高Dice和Point、但尚未完成student阶段的机制。** 固定seed32/40/41各`1/3`的raw-logit ensemble在validation相对seed32为`+0.005829 Scan / +0.004418 Subject / +1 Point`，88/108 scans和41/50 subjects改善。限制是Subject CI下界`-0.005660`低于冻结`-0.003`门，所以旧协议没有授权蒸馏。缺少fresh train-only split上的独立共识复现及单student因果结果。
4. **单轨迹优化会交换Dice和Point，未解决稳定性。** GCSAM fit864在reused test只比Full972 KD高`0.000244` Scan，Subject近乎0差，Point少10；hACE则Point多3但validation Dice下降约`0.0105/0.0114`。这要求下一方法同时冻结Scan/Subject/Point，而不是继续单独优化其中一个。
5. **Test不能导航新配方。** 本实验不读取GCSAM/test逐病例输出，不用test选teacher、权重、epoch、阈值或后处理。三seed数量与`1/3`公式来自2026-08-25旧预注册，下一证据从全新train-only inner开始。

本实验采用第3条启发：重新在fresh fit864上训练三位从未见过inner的独立seed老师，再用严格计算匹配的单老师/三老师student比较，测试“共识能否变成1×单模型”。

## 1. 现在的问题是什么？

Reused internal test270的最高Scan点估计是fit864 GCSAM `0.752003`，但Point只有252/270；最高Subject/Point是Full972 voxel-logit knowledge distillation（逐体素logit知识蒸馏，简称voxel-KD）`0.737327/262/270`，Scan只低`0.000244`。单条随机训练轨迹不能同时稳定占优。

旧三seed等权logit ensemble在fixed validation108达到`0.745841/0.735684/104/108`，同时高于seed32的`0.740013/0.731266/103/108`，但它需要3×推理，且旧CI门阻止了蒸馏阶段。当前真正未回答的问题是：在不挑seed、不调权重的前提下，三位老师的固定共识能否教出一个比单老师student更强、更稳的1×模型。

## 2. 一句话直觉

三位独立老师会在不同题目上犯错；让一个新学生学习三人的平均答案，目标是把“多数人的共识”装进一个脑子，而不是部署三个人一起答题。

## 3. 为什么选择这个idea？

- 2025 ensemble distillation工作把多个独立分割模型的聚合预测蒸馏进单student，说明共识可在降低推理成本的同时保留分割/校准信息。
- 2026脑肿瘤MTSS-KDNet工作与官方代码直接证明多老师单学生可以汇集互补的肿瘤分割知识；论文使用population-specific老师、输出与latent监督，本项目只借“多老师互补→单student”的机制。
- 2026 3D MRI/CT工作给出与项目已成功使用的logit-KD公式和`temperature=2, weight=0.3`，支持保持现有逐体素Kullback–Leibler divergence（KL散度）协议。
- 项目自身证据比论文更直接：三seed等权ensemble同时提高Scan、Subject和Point；旧history/EMA ensemble也成功蒸馏成1×KD200。
- Cross-seed parameter soup失败的是参数平均；本方法只平均冻结模型的输出logit，不要求权重位于同一盆地。
- ReCo、DCD、KAS与LoMix负结果增加了region/feature/self/deep监督；本方法不加新student模块或feature loss，只改变teacher来源，因此它们没有否定本方法。

Primary sources:

- 2025 ensemble distillation: https://arxiv.org/abs/2509.11689
- 2026 multi-teacher brain-tumor distillation: https://www.nature.com/articles/s41598-026-35627-x
- Official MTSS-KDNet code: https://github.com/AhmeddEmad7/Brain-Tumor-Segmentation-Advancing-Generalizability
- 2026 3D MRI/CT logit-KD: https://www.nature.com/articles/s44401-026-00108-w

来源边界：2025论文主要是2D retinal vessel，2026 MTSS-KDNet是多模态/多肿瘤population老师并含latent loss；本项目是同数据、同架构、不同seed的3D T1ce/WT output-only适配，不能称任一论文复现。

## 4. 具体怎么做？

### 阶段A：fresh三seed老师与共识门

从972个development-train病例按subject新建exact 108-scan train-only inner，剩余864 scans为fit。三个teacher seed分别为`20261031/20261032/20261033`，都在fit864从随机初始化训练到固定epoch168；每轮请求972、实际消费968、121步。模型、history/EMA旧teacher、Dice加0.1交叉熵（cross entropy，CE）、`0.3×T²×voxel-KL`、`T=2`、增强、优化器和schedule完全相同，只改变seed。

三位老师训练完成后一次性在fresh inner108评估：三个单模各自raw logit，以及冻结的`z_consensus=(z31+z32+z33)/3`。不按inner剔除seed、改变权重或选epoch。

### 阶段B：计算匹配的单老师student vs 三老师student

只有阶段A全部门通过才运行。两臂student使用同一seed`20261034`、完全相同初始state、fit864、sampler index和168轮协议。

- `single_teacher_control`：每个batch让anchor teacher seed20261031真实前传3次，再把三份相同raw logits平均。
- `three_teacher_candidate`：每个batch让seed20261031/32/33各前传1次，把三份raw logits按`1/3`平均。

两臂teacher forward都恰为3次，计算、随机数消费和student更新次数匹配。Student输入同一增强T1ce；teacher全部eval/frozen且看同一增强输入；student loss固定为真实标签DiceCE加`0.3×T²×binary voxel-KL`、`T=2`。唯一差异是三份teacher logits来自同一模型还是三个独立seed。部署只保存普通4.70M student。

### 阶段C：fresh-seed Full972 final

只有阶段B全部门通过才运行。两臂在full972用fresh student seed`20261035`从相同随机state训练到epoch168：

- Control把已有Full972 seed20260932 teacher真实前传3次。
- Candidate让已有Full972 seed20260932/40/41各前传1次。

完成并冻结两个checkpoint后，才一次性访问fixed validation108。Test不自动授权。

## 5. 唯一变化与固定项

- 阶段A唯一变化：teacher随机seed；共识公式预先固定为raw logits各`1/3`。
- 阶段B/C唯一arm差异：三个teacher forward来自同一anchor的重复输出，还是三个独立seed的输出。
- 固定：T1ce→WT `{1,2,3}`、image-only brain crop、true-128³ cache、4,700,897参数width16 SegResNet、dropout0.1、student seed/initial state、batch8、native-volume weighted replacement、972 requested/968 consumed/121 steps、DiceCE、voxel-KD weight0.3、temperature2、AdamW、late-cosine 200轮分母/固定停168、shift8、clip5、threshold0、无test-time augmentation（测试时增强，TTA）、无Segment Anything Model（SAM）、无后处理、1×student部署。
- 新split seed`20261030`；teacher seeds`20261031/32/33`；inner student seed`20261034`；final student seed`20261035`。

## 6. 成功、失败和停止规则

### 阶段A

- 固定评测epoch168，不选epoch。
- 等权ensemble必须严格超过三个单模各自的Scan与Subject；Point至少等于三个单模中的最高值。
- Ensemble相对三个单模的Scan/Subject 10,000次paired bootstrap 95% CI下界全部`>=-0.003`。
- 任一门失败则不运行student。

### 阶段B

- Candidate必须严格超过matched Control的Scan与Subject，Point不得下降。
- Scan/Subject paired 95% CI下界都`>=-0.003`。
- 初始state、全部epoch sampler index、首批增强tensor、teacher forward数、student loss/update必须exact配对。
- 任一门失败则不运行Full972 final。

### 阶段C

- Candidate必须同时严格超过fresh matched Control与archived Full972 seed32 validation `0.740012535/0.731266217`的Scan/Subject，Point至少103/108。
- 相对两个comparator的Scan/Subject四个paired 95% CI下界全部`>=-0.003`。
- 全部工程、hash、reload、time、memory与access门通过。

### 通用停止与test边界

- Manifest/cache/native artifact/checkpoint/document SHA不符；fit/inner case或subject泄漏；teacher有梯度；任何loss/gradient非有限；teacher forward数不等；pairing hash不一致；batch8 peak allocated>24GiB；单臂训练>6h；reload/deploy logit不exact，立即停止。
- 失败后不换seed、不剔除teacher、不调`1/3`权重、KD weight、temperature、epoch、LR、threshold或postprocess。
- 即使Stage C validation全过，也只产生test候选；reused test270必须另写pre-access receipt并得到用户再次明确授权。

## 7. 成本与风险

- Stage A：三位fit864老师，每个约3.5–4h；两卡先并行两位、再跑第三位，预计双卡墙钟7–8h。
- Stage B：paired students预计4–5h双卡墙钟。
- Stage C（仅过门）：paired Full972 students预计4–5h。
- 显存：预计每卡20–21GiB，硬门24GiB。三个teacher顺序no-grad forward，不同时保留activation graph。
- 磁盘：复用现有约9.5GB train cache；split、7个左右checkpoint和报告预计<300MB，不建新cache。当前约15GB可用。
- 推理：最终仍4,700,897参数、1×locator forward；无teacher、ensemble、TTA、SAM或postprocess。
- 最大风险：旧三seedensemble的正点估计可能不能在fresh split复现；即使ensemble复现，同容量student也可能只能吸收部分共识，或Dice提高但Point不稳。

## 8. 最终结果（完成后填写）

- Stage A fresh teachers/ensemble：待运行。
- Stage B paired students：仅Stage A全过门后运行。
- Stage C Full972 final/validation：仅Stage B全过门后运行。
- Test：未授权。
- 置信区间/病例分布：待运行。
- 是否支持原直觉：待运行。
- 学到了什么：待运行。
- 报告/归档/哈希：待运行。

