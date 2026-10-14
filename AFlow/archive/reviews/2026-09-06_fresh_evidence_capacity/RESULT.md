# 新内部证据容量审计：immutable role下仅剩2人/3 scans

状态：✅ 只读manifest复算完成；没有图像/标签体素、模型或指标重算。

## 结论

当前972个train-pool scans属于444位患者。严格沿用no-N4 E/F建立时的规则，把所有历史inner、selector和confirmation患者永久视为已揭示，合并排除406人。E/F选择之前剩38人/87 scans；E42/17人与F42/19人互斥，共使用36人/84 scans。现在仅剩：

- `BraTS-GLI-02633`：1 scan；
- `BraTS-GLI-02870`：2 scans。

合计2 subjects / 3 scans，无法建立一个与既有最小规模相当的42-scan train-only inner，更不可能两折复制。

## 这意味着什么

在当前“数据角色不可静默重分配”的证据政策下：

- 已揭示E/F不能评新方法；
- 历史fit患者不能事后变成validation而假装fresh；
- 3 scans不能承担Dice/Point promotion gate；
- 因此本30小时窗口不应再启动新的内部单因素训练，否则只有训练loss，没有合法的新泛化证据。

这不是说仓库永远不能做内部研究。若用户未来明确改变目标，可预先声明role reassignment并从零重训所有涉及模型/teacher，但证据必须标为reused development，不能称fresh；更好的选择是取得新的人工标签队列或扩展正式训练数据后另建未见fold。

## 证据

- [机器审计](CAPACITY_AUDIT.json)，SHA-256 `72fb9f182cd1addceb7e61234e4447272ea73d5dfc4bb92323d9631e6b02c5a9`
- Frozen Full972 manifest SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`
- E/F split SHA-256 `68975292cc10ab581657a97789050a28cc03d4a3bc3b0991682b7284bfa9c3f2`

访问：manifest metadata only；image/label voxel=0，model forward=0，internal test270/fixed validation/outer/confirmation/UCSD/Africa95新访问=0。

## 关联的冻结cache域诊断

另对已有train972、UCSD official75、Africa95三份128³ cache做只读全量统计，没有模型或新指标访问。Africa WT均值/中位数是train的`2.87×/4.09×`、UCSD的`2.47×/3.38×`；Africa输入非零占比、全体素std和mean-absolute intensity也整体右移。标签范围、输入支持、治疗阶段、采集和重采样同时变化，不能据此定位单一因果或调阈值。

证据：[机器统计](INPUT_DOMAIN_SHIFT_AUDIT.json)、[ECDF图](INPUT_DOMAIN_SHIFT_AUDIT.png)，SHA-256 `fb0eeea59ee5bae5f4bb8285fc8c389c1cbe8b39e0e5f460603dfd06e88ab6c1` / `6fa4cb1f7d5a82eb319684bd11f008c53162502b2c28a933e16bd204cb3ae2a8`。

WAW/CPTAC完成后，V2把五份cache并排而不改V1：相对train972，UCSD/Africa/CPTAC/WAW的WT mean比=`1.16/2.87/2.13/1.47`，median比=`1.21/4.09/2.89/1.72`；全体素std median差=`+0.014/+0.052/+0.025/+0.019`，非零占比median差=`+0.020/+0.038/+0.016/+0.013`。Africa在三轴均最极端；WAW是中度偏移，CPTAC虽target更大却绝对Dice最高，直接说明target体积不能单独解释准确率。证据：[V2机器统计](INPUT_DOMAIN_SHIFT_AUDIT_V2.json)、[V2图](INPUT_DOMAIN_SHIFT_AUDIT_V2.png)，SHA-256 `a9fab522fb1a98a2a13053f7ecb4e9e5022703327bef604df108eade626f7a44` / `2edfb2306febe72d445780b3e206b02562e6f735093feddf891d08ff29aaee19`。

WAW-pre cache回收后不伪造逐例CDF；postop addendum只读V2已归档preop汇总和当前postop31 cache。Postop相对preop target mean/median=`1.017×/1.071×`，all-voxel std median仅`+0.000334`，nonzero fraction median仅`+0.00249`。三个粗域指标近似，但Point约0.46 vs preop约0.92–0.94且miss集中RC，说明简单体积/全局尺度不能解释早期术后失败。两组患者不同、grade分布不同，不能作术前→术后因果。[Addendum](WAW_POSTOP31_DOMAIN_ADDENDUM.json)、[图](WAW_POSTOP31_DOMAIN_ADDENDUM.png)，SHA `739bca1c…` / `41fca0fd…`。

训练采样公式本身也经只读复算：inverse-sqrt native-WT权重使最小/最大体积四分位分别占44.0%/12.7%概率，每例每轮预期1.75/0.506次；最大四分位单病例每轮平均60.4%概率不被抽到。这符合历史“优先小病灶恢复Point”的设计，可能与Africa大型WT绝对漏分方向相容；但Full/GCSAM都用该公式，不能解释二者差值，也不能由外部结果改权重。[采样暴露审计](TRAIN_SAMPLING_EXPOSURE_AUDIT.json)。
