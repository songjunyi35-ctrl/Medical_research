# Full972 KD168 与 GCSAM184 跨cohort冻结证据综合

状态：只读已有raw报告完成；0新模型forward、0阈值/病例/模型选择。

## 六个cohort必须分开看

| cohort | 证据身份 | Full identity Scan | GCSAM identity Scan | GCSAM−Full paired 95% CI | Point Full→GCSAM |
|---|---|---:|---:|---:|---:|
| internal test270 | reused、非盲、非外部 | `0.751760` | `0.752003` | `+0.000244` `[-0.005655,+0.005740]` | `262→252` |
| UCSD75 official-float | 同75例来源纠正、reused public external | `0.746255` | `0.755890` | `+0.009634` `[-0.002163,+0.025053]` | `70→70` |
| BraTS-Africa95 | public post-challenge、四中心机构独立/no patient bridge | `0.523003` | `0.591619` | `+0.068617` `[+0.044472,+0.092498]` | `87→84` |
| CPTAC33 public Training | Africa-informed public confirmation、非hidden blind | `0.841331` | `0.843285` | `+0.001954` `[-0.006646,+0.012384]` | `33→33` |
| WAW-GliomAI152 | public preop、institution-disjoint/no patient bridge、非hidden blind | `0.683378` | `0.693601` | `+0.010223` `[-0.003005,+0.023481]` | `141→140` |
| WAW early-postop31 | preop患者零重叠、day0–22、public非hidden blind | `0.591647` | `0.593305` | `+0.001657` `[-0.012498,+0.016205]` | `15→14` |

加入CPTAC33、WAW preop152与WAW early-postop31后，GCSAM−Full的Scan差从Africa的`+0.068617`回到`+0.001954/+0.010223/+0.001657`，三区间均跨0。Point差为`−10/0/−3/0/−1/−1`。因此旧“internal→UCSD→Africa递增”只是一段已见描述；三个后续队列都不支持单调或普适规律。Africa95仍是真实的队列特异checkpoint优势。

## 不能推出什么

- 六个cohort的患者、治疗阶段、标签语义、医院、设备和grid不同，不能把它们合并成一个排名或回归。CPTAC33、WAW preop与postop共同反驳“域越远必然越适合GCSAM”的单调叙事。
- Africa95方向不能推翻GCSAM fixed184双内部复制失败；它支持单个冻结checkpoint在这个域的coverage，不证明优化器配方可复制。
- 不能据图选择checkpoint、`rho`、阈值、TTA或体积router。需要新的未见队列验证任何新假设。
- UCSD official-float是同已访问75例的来源纠正，不等于第二个独立外部点。
- CPTAC33是public Training labels、external-to-project但不是hidden blind；其source-disjoint无患者ID桥，且两checkpoint仍共享训练源parity caveat。
- WAW152是public preop delayed T1c，institution-disjoint但无患者ID桥、非hidden blind；发布者预处理、阶段和标签语义也是共同变化，不能单因解释。
- WAW postop31与preop已见患者ID零重叠，但同Warsaw source且只覆盖day0–22；低Point主要落GT切除腔，仍不能跨阶段作因果比较。

## 证据

- V4（6 cohort）[机器结果](RESULT_V4.json)，SHA-256 `966bfc7df99dad1951b47ee51fb7504521bc765acd1bf8663a8e342ed54725da`
- V4 [图](COMPARISON_V4.png)，SHA-256 `a8e9881e4f2fbc2dcf02e1ce7e75e09976af9d093c09fd7993f7eb5988eb8012`
- V4 [PDF](COMPARISON_V4.pdf)，SHA-256 `ce17731a81a6bf7d54c45ef2d624a91197ac49f9063f277636303b6be25b934e`
- V3（postop访问前）[机器结果](RESULT_V3.json)，SHA-256 `206bca7360e06908f5993d90a29b301e120a22646093be5c8af1d7c281b09ace`；[图](COMPARISON_V3.png)，SHA-256 `26dad06059d492e6da2f1dd5e93a0afa3cfaa8189f791fa75d0db312365782a0`
- V2（WAW访问前）[机器结果](RESULT_V2.json)，SHA-256 `43bff40dde122fa15ed8d85969b2fedd06ad2d39322f30ce2959727ae2d033fd`；[图](COMPARISON_V2.png)，SHA-256 `fd288c90c3bb9c0c200968b24be473de42909ab64ca67b9089d528ac4e56c9dc`
- V1（CPTAC访问前）[机器结果](RESULT.json)，SHA-256 `9027dd405bf8eb20f6e30a49ddc58dc32914f7f54eb9cc4f1658e174246302fb`；[图](COMPARISON.png)，SHA-256 `d19e2191a653c175645ae3ac43e12f7d14e3f6b6130fa426d016a9d6a213fbd0`
- internal raw SHA `d0e2f48e1f499d65adddb3dbe30758caad27768bb75f737ea2cfe4a877c335d5`
- UCSD raw SHA `a1155914b16b648f97f11a7d7a20cb27096d7eed9f9662db73f4905860da04d2`
- Africa raw SHA `c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3`
- CPTAC raw SHA `4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd`
- WAW raw SHA `ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91`
- WAW postop raw SHA `72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98`

## V5补记（2026-09-06 Fable session）

加入第七个cohort UCSD-BraTS2024test56（56 scans/41 subjects，public post-challenge BraTS-GLI 2024 test的UCSD部分，patient-disjoint by official mapping、非institution-disjoint）。GCSAM−Full identity Scan差在七cohort为：internal `+0.000244`、UCSD75 `+0.009634`、Africa95 `+0.068617`、CPTAC33 `+0.001954`、WAW-pre `+0.010223`、WAW-post `+0.001657`、**UCSD-BraTS2024test56 `−0.012312`（CI `[-0.025822,-0.000092]`，第一个整体为负的区间）**；Point差=`−10/0/−3/0/−1/−1/−4`。方向依cohort可逆："GCSAM外部普遍偏高（仅Africa确定）"措辞作废，改为"Africa确定为正、UCSD-BraTS2024test56确定为负、其余五处不确定"。0新forward；不pool、不回流选择。V5 JSON/PNG/PDF SHA=`19d842c4…/4776ac1b…/a03cb2f5…`；见[RESULT_V5.json](RESULT_V5.json)与[COMPARISON_V5.png](COMPARISON_V5.png)。
