# Fresh三seed共识蒸馏结果

## 结论

Fresh三seed等权raw-logit共识在train-only inner108上的Scan、Subject和Point点估计都不低于三个单模，但没有稳定超过最强seed20261031。两个相对强seed的CI安全门失败，因此Stage A停止，`student_stage_authorized=false`；没有训练student，没有运行Full972 final，也没有访问fixed validation108、outer、confirmation或test270。

通俗地说：三人平均能明显救两位较弱老师，却几乎没有超过这次最强老师。旧三seedensemble的明显收益没有在全新split上稳定复现，因此现在蒸馏只是在压缩一个“并未明确更强”的teacher，不值得继续花GPU。

## 方法与唯一变化

- 从Full972 development-train按subject生成fresh split：864 scans / 401 subjects fit；108 scans / 43 subjects train-only inner；case/subject零重叠。
- 在fit864上以seed`20261031/32/33`训练三个4.70M plain SegResNet到固定epoch168。三者数据、每轮968 effective draws/121 steps、history/EMA voxel-KD teacher、DiceCE、KD weight0.3、temperature2、增强、AdamW和LR schedule相同，只改变seed。
- 三个checkpoint全部冻结后，一次性在fresh inner比较三个单模raw logit与固定`(z31+z32+z33)/3`。不选择epoch、不剔除seed、不调权重、阈值或后处理。

## 实现与工程过程

- Split builder第一次因仓库根模块路径缺失在输出前停止；第二次生成两个role JSONL后因Python `false/False`拼写在manifest前停止。Partial v1目录保留，方法/seed不变后在v2生成权威split。
- 共识exact mean、三输入/shape和finite合同3项通过。
- 三个teacher各2-step真实GPU smoke通过，初始state SHA互不相同；peak均约20.29GiB。
- 三个正式teacher均完整168轮、checkpoint reload exact、退出码0；没有inner/validation/test loader。

## Fresh train-only inner108 / 43 subjects

| 方法 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| seed20261031 | `0.746411714` | `0.735066792` | **`106/108`** |
| seed20261032 | `0.736108619` | `0.722151276` | `104/108` |
| seed20261033 | `0.732792015` | `0.718893574` | `103/108` |
| **固定1/3共识** | **`0.746942195`** | **`0.735129559`** | **`106/108`** |

共识相对seed20261031：

- Scan `+0.000530481`，95% CI `[-0.009103203,+0.007084167]`；82/108 scans改善、26变差。
- Subject `+0.000062767`，95% CI `[-0.016104621,+0.009852094]`；34/43 subjects改善、9变差。
- Point持平106/108。

共识相对seed20261032：

- Scan `+0.010833576`，95% CI `[+0.004547324,+0.018530646]`。
- Subject `+0.012978283`，95% CI `[+0.004962369,+0.022912493]`。
- Point`+2/108`。

共识相对seed20261033：

- Scan `+0.014150180`，95% CI `[+0.009873257,+0.019694893]`。
- Subject `+0.016235985`，95% CI `[+0.010841622,+0.022971883]`。
- Point`+3/108`。

## Promotion与科学解释

通过：共识严格超过三个单模Scan/Subject点估计；Point等于最佳单模；相对seed32/33的四项Dice CI门全部通过；全部工程门通过。

失败：

- 共识相对最强seed31 Scan CI下界`-0.009103<-0.003`。
- 共识相对最强seed31 Subject CI下界`-0.016105<-0.003`。

因此`student_stage_authorized=false`。

原直觉只得到部分支持：独立seed确实互补，等权平均能大幅修复较弱轨迹；但共识是否超过一个幸运/强轨迹取决于seed组合，本次对强seed的平均收益几乎为零且tail波动很大。它没有解决“强单模不可稳定复现”的核心问题。

这个结果只否定当前fresh split、三个冻结seed、equal raw-logit mean的完整晋级主张；不证明所有multi-teacher KD无效。但本inner已揭示，不能据此剔除seed31、重配权重或设计adaptive teacher gate后继续在同一inner验证。

## 成本

- seed31/32/33 formal：`12,225.41s / 12,162.45s / 12,234.31s`，各约3.38–3.40h。
- 两卡先并行seed31/32，再跑seed33；Stage A总墙钟约6.79h。
- 每个arm peak allocated `20.292GiB`。
- 复用现有cache；新增三个checkpoint与报告，未创建大cache。
- Student未训练；部署成本未改变。

## Evidence与访问

- Raw inner report SHA-256 `33fe034e829cf7498cd35083c12f3181a47d5db7eabd753f482612edcf15ef5d`。
- Inner started receipt SHA-256 `c469d257ea68fcca04a7f432f0e512b1969a4e50ece2e363cfce4902bdc74067`。
- Pre-access receipt SHA-256 `c44534afe12a3898fe187340f579e509dd9997245aefaeb98577bfa28dfcccc4`。
- Teacher checkpoint SHA-256：seed31 `ef2bfeaf4ce3ca87990d1af1b298206fad503858a7dd0f88845a31530e6f8868`；seed32 `40eea2354e5f1059c31d12f635233bb295cc821df1d91ab396388ab892c6e6b9`；seed33 `af634de143cb19ddd404c4a71e35f25af271b1101a69daa2c8cd6dd8fdeb70eb`。
- Fresh train-only inner已访问；fixed validation108、outer、confirmation、test270均未访问。

## 下一决定

停止当前等权三seed共识蒸馏，不进入student/Full972 final，不剔除seed、不调teacher权重。当前本地没有额外未揭示BraTS/外部cohort；下一方法若继续研究稳定性，必须建立另一套fresh证据，不能复用本inner做adaptive teacher选择。

