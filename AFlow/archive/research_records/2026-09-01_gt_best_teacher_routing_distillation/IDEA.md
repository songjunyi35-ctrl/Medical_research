# Experiment idea: 训练标签引导的每病例最佳老师蒸馏

Status: planned and frozen before implementation, GPU model construction, training, or fixed-validation access.

## 链路审视与启发

1. Full972数据从随机初始化参与仍是最大成功主线，相对matched Full756 validation为`+0.010943/+0.007982`；width24/160³负结果要求下一步不增加student容量或分辨率。
2. Fresh三seed等权共识能显著救弱seed32/33，但相对强seed31只有Scan/Subject `+0.000530/+0.000063`，95% CI下界`-0.009103/-0.016105`。82/108 scans改善，仍有26 scans被平均伤害。固定平均不知道每例该听哪位老师。
3. GCSAM与hACE再次证明Dice和Point会交换：GCSAM reused test Scan只`+0.000244`但Point少10；hACE Point增加而validation Dice下降约`0.0105/0.0114`。新方法必须同时冻结Scan/Subject/Point。
4. 本地没有额外未揭示BraTS/外部cohort。当前fresh inner已用于提出本假设，不能继续在其上验证新routing；因此新学生直接按冻结full972协议训练，唯一一次开发评估使用fixed validation108并明确标记reused development evidence。
5. Test270不参与teacher选择、routing、公式、seed或门槛；test仍未授权。

## 1. 现在的问题是什么？

三个独立seed的错误确实互补，但固定等权平均的收益取决于seed组合：在fresh inner上，共识相对弱seed可提高Scan/Subject约`0.0108–0.0162`，却几乎没有超过强seed31。平均时，弱老师也会把强老师本来正确的病例拉坏。

目标是让1×student吸收“每个训练病例上最可靠的teacher”，而不是吸收固定平均。当前Full972已有seed20260932/40/41三个冻结老师，无需再训练teacher。

## 2. 一句话直觉

三位医生各有擅长病例；训练时我们手里有标准答案，就让student对每个病例听分得最准的那位，而不是把三份意见无差别平均。

## 3. 为什么选择这个idea？

- 2025 adaptive weighted multi-teacher medical segmentation论文使用ground-truth Dice loss为每个batch动态计算teacher权重，证明GT引导teacher贡献是可行机制。
- 2026 MTSS-KDNet脑肿瘤工作按已知population标签选择匹配teacher，说明训练期teacher routing可以把专长压入单student，部署时不保留router或teacher。
- 本项目fresh共识结果给出直接动机：弱seed在部分病例提供明显互补，但固定平均对强seed有负尾。
- Parameter soup失败的是权重平均；本方法选择output teacher。DCD/ReCo/KAS失败的是额外feature/region/self loss；本方法保持现有output-only voxel-KD。
- 这不是2025论文的忠实复现。该论文按Dice loss正比给更差teacher更高权重，且没有充分消融权重方向；本项目在任何新validation前冻结更直接的hard-best规则：每病例选择soft Dice loss最低的teacher，无可学习router、无权重温度、无超参搜索。

Sources:

- 2025 adaptive multi-teacher KD: https://doi.org/10.1016/j.knosys.2025.113196
- Open-access paper copy: https://repositori.urv.cat/repositori/getDocument/imarina%3A9452339?ds=DocumentPrincipal&mime=application%2Fpdf
- 2026 brain-tumor MTSS-KDNet: https://www.nature.com/articles/s41598-026-35627-x
- Official MTSS-KDNet code: https://github.com/AhmeddEmad7/Brain-Tumor-Segmentation-Advancing-Generalizability

## 4. 具体怎么做？

两臂都在Full972上用fresh student seed`20261040`从相同随机initial state训练到固定epoch168。每轮请求972、实际消费968、121步。三次teacher forward全部eval/no-grad，输入与student使用同一增强后的T1ce。

### Control：单anchor重复三次

冻结Full972 seed20260932 teacher真实前传3次，得到三份完全相同的raw logit；teacher target就是anchor logit。重复三次是为了与Candidate匹配teacher计算次数、显存行为和调用合同。

### Candidate：每病例GT最佳teacher

冻结Full972 seed20260932/40/41各前传1次。对batch中每个病例、每位teacher，先把raw logit经过sigmoid得到前景概率，再与ground truth（GT，真实标签）计算soft Dice loss：

`L_j = 1 - (2 * sum(p_j*y) + 1e-6) / (sum(p_j) + sum(y) + 1e-6)`。

每病例选择`argmin_j L_j`，把该teacher的整幅raw logit作为student的teacher target。选择与teacher logits全部detach，不向teacher反传。

两臂student loss完全相同：真实标签Dice + 0.1交叉熵，再加`0.3*T²*binary voxel KL`，temperature`T=2`。最终只部署普通4.70M student、raw logit阈值0、argmax作为自动Point。

## 5. 唯一变化与固定项

- 唯一arm差异：teacher target来自固定anchor，还是按每病例GT soft-Dice从三个seed中hard选择最佳teacher。
- 固定：Full972 manifest/cache/native-volume sampling、student seed20261040与initial state、batch8、每轮968 draws/121 steps、增强随机流、4.70M width16 SegResNet、dropout0.1、DiceCE、voxel-KD weight0.3、temperature2、AdamW、LR schedule、固定epoch168、clip5、teacher forward count3、threshold0、无TTA/SAM/postprocess、1×部署。
- Control/Candidate分别在两张A100独立进程并行，不使用DDP。

## 6. 成功、失败和停止规则

- 两臂必须完整168轮，initial state、每epoch sampler indices、首批增强image/label、student输入和更新合同exact。
- Candidate routing必须每病例恰选一个teacher；三teacher forward数必须为3；teacher无gradient；记录每epoch/全程routing counts但不作为选择门。
- 训练完成后冻结runner/checkpoint/report/cache/manifest/validation evaluator/阈值/bootstrap哈希，再一次性访问fixed validation108。
- Candidate必须同时严格超过fresh matched Control和archived Full972 seed32 validation `0.740012535/0.731266217`的Scan与Subject。
- Candidate Point至少103/108。
- Candidate相对Control与archived seed32的四个paired Scan/Subject 10,000-bootstrap 95% CI下界全部`>=-0.003`，bootstrap seed`20261041`。
- 任一门失败停止，不调routing规则、seed、teacher成员、KD weight、temperature、epoch、LR、threshold或postprocess。
- 即使validation全过，也只产生test候选；reused test270必须另写收据并得到用户再次明确授权。

## 7. 成本与风险

- 两臂都恰有3次teacher forward，预计双卡墙钟4–5h；单臂硬门6h。
- Peak预计20–21GiB/卡，硬门24GiB。
- 复用现有cache与三个Full972 teacher checkpoint；新增两个student checkpoint和报告预计<150MB，不建新cache。当前约15GB可用。
- 推理仍4,700,897参数、1×forward，无teacher/router/ensemble/TTA/SAM/postprocess。
- 最大风险：训练GT选择可能在不同病例间频繁切换teacher，student难以形成一致函数；或者oracle routing只在训练病例有效，validation无法泛化。

## 8. 最终结果（完成后填写）

- Formal training：待运行。
- Fixed validation：仅formal与pre-access全过门后运行。
- Test：未授权。
- Routing分布：待运行。
- 是否支持直觉：待运行。
- 报告/归档/哈希：待运行。

