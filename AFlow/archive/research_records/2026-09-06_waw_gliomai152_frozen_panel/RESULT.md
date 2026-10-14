# WAW-GliomAI152冻结checkpoint面板：最终结果

状态：一次性public external-to-project评测完成并关闭；不是hidden blind，不调参、不重跑。

## 初学者结论

这次把两个已经冻结的自动分割模型带到华沙医科大学152名术前胶质瘤患者上，只看T1对比增强MRI，把人工标注的坏死核心、FLAIR异常和强化肿瘤全部合成全肿瘤（WT）。GCSAM的一次推理与Full的八镜像平均Dice几乎相同：`0.693601`对`0.693494`，差仅`0.000107`；前者计算便宜，后者最高点落入肿瘤的病例更多。因此没有一个模型同时赢Dice、Point和成本。

## 只改变什么，哪些不变

唯一新增是WAW-GliomAI final release的152名术前患者。发布描述称156份术前检查，但最终coreg T1c与preop mask严格交集只有152，模型前就按最终文件冻结。模型、checkpoint、canonical RAS/N4 shrink2、image-only crop、非零z-score、128³、WT=`{1,2,3}`、阈值0、Full固定八镜像raw-logit等权平均、GCSAM identity、无Segment Anything Model（SAM）/后处理/模型ensemble，以及10,000次bootstrap均不变。

## 绝对结果

152 scans=152 subjects，每人一张，因此Scan与Subject Dice相同。

| 方法 | Scan=Subject Dice（95% CI） | Point hit | forwards |
|---|---:|---:|---:|
| GCSAM184 identity | **`0.693601158`** `[0.660656,0.723433]` | 140/152 (`0.921053`) | `1×` |
| Full972 KD168固定Flip8 | `0.693494112` `[0.660273,0.723615]` | **143/152 (`0.940789`)** | `8×` |
| Full972 KD168 identity | `0.683377992` `[0.649811,0.713659]` | 141/152 (`0.927632`) | `1×` |

“总体Dice点估计最高”和“单模型1× Dice点估计最高”在WAW152上都是GCSAM，但它只比Full Flip8高`0.000107046`；没有预注册两者的直接优越性检验，且Point少3，所以不能称三指标冠军。

## 配对结果与直觉

- GCSAM identity−Full identity：Dice=`+0.010223166`，paired 95% CI=`[-0.003005,+0.023481]`，84改善/63变差/5 tie，Point=`−1`。按冻结门是`positive_but_uncertain`：方向偏正，但区间仍容许零或小幅反向。Africa95的`+0.068617`确定优势没有在CPTAC33和WAW152稳定复现，因此只保留为Africa队列特异checkpoint现象，不是跨域普适规律，更不是GCSAM优化器配方因果证据。
- Full Flip8−identity：Dice=`+0.010116120`，paired 95% CI=`[+0.004472,+0.016274]`，93改善/54变差/5 tie，Point=`+2`。按冻结门是`supported_on_WAW`。连同reused internal270和CPTAC33，这是第三处Scan配对区间支持；Africa95仍跨0。直觉得到跨队列方向支持，但真实部署代价仍约8×。

## 工程失败边界与来源可信度

第一次正式命令因预访问SHA字段位于`code.evaluator_sha256`而评测器读取顶层字段，发生KeyError；错误在CUDA device、started receipt、checkpoint和model之前，0 forward/0性能。失败与唯一字段路径修复完整保留，V2预访问重新绑定新评测器SHA后才进行唯一正式访问。

官方大包3,164,137,716 bytes与MD5 `0d24a3a2c76bc6fa9b5b6468679676b1` exact，zip CRC全过；152份选中T1c共785,599,858 bytes。终态独立重算官方大包、304个选中图/标签文件共789,646,100 bytes、两个checkpoint、1.48 GiB cache、三绝对/两配对bootstrap和三行榜单，全过。

## 事后误差机制（只解释，不选择）

从raw保存的intersection/predicted/target计数离线重算：Full identity/Flip8/GCSAM的平均precision=`0.8006/0.8425/0.7987`，recall=`0.6345/0.6356/0.6413`。这里沿项目既有约定把“预测为空”时precision记为1；Flip8−Full precision=`+0.04185`，事后CI `[+0.01957,+0.06881]`全正；recall=`+0.00110`，CI `[-0.00413,+0.00600]`跨0，平均少预测763 voxels。故WAW上的Flip8收益主要伴随减少多画，而不是确定补回漏画。旧V1误把空预测precision记为0，已由V2替代；该口径修正不改变Dice、Point、recall、预测体积或正式门结论。

GCSAM−Full precision=`−0.00190`、CI跨0；recall=`+0.00674`、CI也跨0，平均多预测1,299 voxels，不能把`+0.0102`确定归因于coverage。真实WT在train972经验分位中位68.31%，44/152高于train p90、24/152高于p95、0超过最大，比Africa温和。按真实体积四分位，GCSAM差=`+0.0114/+0.00134/+0.00093/+0.0272`，但log体积相关ρ仅`0.125`、名义p=`0.125`；WHO grade2/3/4差=`+0.0289/−0.00555/+0.00636`。这些是已见分组、与scanner/病理/体积共变，不可变成router或阈值规则。[机器审计V2](POSTHOC_ERROR_AUDIT_V2.json)、[图V2](POSTHOC_ERROR_AUDIT_V2.png)。

尾部安全审计显示Full identity/Flip8/GCSAM分别有5/6/5例Dice=0、20/19/18例Dice<0.5。三者共同5个零Dice全部是WHO grade2；其中4例mask只有FLAIR abnormality label2、没有enhancing label3或necrotic label1，说明T1对比增强单模态被要求预测主要由FLAIR定义的区域时存在可观测性限制。123例有label3时三模型平均Dice约`0.701/0.711/0.711`；29例无label3时约`0.611/0.620/0.620`。这是事后标签组成描述，绝不能用真值label/grade选择模型、排例或路由。[失败尾部](POSTHOC_FAILURE_TAIL.json)。

Train-only标签审计又显示：训练并非没有无label3扫描（269/972），但纯only-label2仅30/972 scans、18/444 subjects；WAW为26/152，比例17.11% vs 3.09%，约`5.54×`。Train多数无label3仍有label1或切除腔label4提供结构线索，WAW术前没有label4。该覆盖差与零Dice相容但非因果，且无fresh-inner容量验证新采样法。[可观测性审计](../../reviews/2026-09-06_t1ce_observability_audit/RESULT.md)。

Scanner描述也限制GCSAM解释：SIGNA Architect占112/152，GCSAM−Full在这组`+0.01365`，其余40例仅`+0.00063`；3T 114例为`+0.01611`，1.5T 33例为`−0.00341`。Flip8在Architect/其他为`+0.01177/+0.00549`，在3T/1.5T为`+0.01306/+0.00526`，方向均正。所有分组都是事后且scanner/场强/grade/label组成共变，不能声称因果或做scanner router。[Scanner描述](POSTHOC_SCANNER_DESCRIPTION.json)。

## 代价、访问和限制

- 训练0；cache 630.7秒；正式1368 forwards、26.49秒、峰值5,457,695,744 bytes（约5.08 GiB）。
- 源大包2.95 GiB；选中T1c约749 MiB；cache约1.48 GiB。Full Flip8参数/checkpoint不增，但推理约8×；GCSAM/Full identity为1×。
- 本实验只访问WAW152；未访问internal test270、fixed validation108、outer、confirmation、UCSD75、Africa95或CPTAC33的图像/cache/模型。
- WAW是public、非盲、institution-disjoint/no-known-overlap；没有跨库patient-ID桥。发布`t1c`是delayed post-contrast，且发布者已做coreg/skull-strip/bias correction后项目又按冻结协议运行N4；这些与术后/治疗后训练语义一起限制横向比较。
- 两checkpoint训练源仍是本地BraTS2024 Kaggle mirror，官方Synapse corrected-v2字节parity未证明。

终态验证与全部cache SHA归档后，为给后续来源保留磁盘，按既有授权先删除派生`cache128` 1,593,969,520 bytes，后又删除可从保留官方preop zip重建的`selected_t1c` 786,230,642 bytes。WAW官方preop大包、raw、结果、代码、checkpoint、逐文件source SHA收据和cache index归档均保留；两个删除目录均可按固定脚本重建，但不再常驻。

## 决定与证据

WAW152关闭，不按结果改病例、modality、N4、阈值、view、权重或checkpoint，也不重测。保留GCSAM作为WAW最高1× Dice点估计，保留Full Flip8作为WAW最高Point与区间支持的8×选项；不同cohort分榜，不宣布通用冠军。

- [七问](IDEA.md)
- [预注册](PREREG.md)
- [raw](RAW_WAW152.json)，SHA-256 `ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91`
- [V2模型前收据](MODEL_PREACCESS_V2.json)，SHA-256 `bea3a59e70fab071b29924d087c2671e87ac34adf281b0942471d8ef2e726858`
- [机器结果](RESULT.json)，SHA-256 `d00da5b606fc0800453f5b7b485b5ec71aba214e3fc0ba313b0ca75559951104`
- [终态验证](VERIFICATION_RECEIPT.json)，SHA-256 `cbe26438822f93cb8a498e7505019d1cd901de4b77c42c84b858274f1288f411`
- [事后误差审计V2](POSTHOC_ERROR_AUDIT_V2.json)，SHA-256 `f7bf7b75fc58068afed10e0649447855c66c7ef701f8d4596bc222df05f40428`
- [事后图V2](POSTHOC_ERROR_AUDIT_V2.png)，SHA-256 `799f68b64fe86b053d7521a9531311e46491a77c7c26f0aa97d84cd6c9bc3aae`
- V1 `POSTHOC_ERROR_AUDIT.json/.png`保留作审计轨迹，但因空预测precision口径错误已被V2取代；正式Dice/Point结果从未改变。
- [失败尾部审计](POSTHOC_FAILURE_TAIL.json)，SHA-256 `cdfed37eb2e7586c65525cc25d7b1bf420e172b63a76798050d4a887675ae714`
- [Scanner/场强描述](POSTHOC_SCANNER_DESCRIPTION.json)，SHA-256 `00a3ed387669892016c38a37cc9f6d4e0f54a2a8d4aa9b4a939a79396fcda0d5`
- [来源审计](../../reviews/2026-09-06_waw_gliomai_source_audit/RESULT.md)
- [Full checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth)，SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`
- [GCSAM checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth)，SHA-256 `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`
