# Full972 同轨迹最近六轮权重平均结果

## 结论

LAWA6（Latest Weight Averaging，最近权重平均）在三个独立fresh seed中都让同一run的Scan Dice和Subject Dice点估计上升，且Point不下降，说明“把同一训练轨迹最后六轮参数取中心”是可复现的局部稳定机制。但第三个fresh Full972 final在fixed validation108上仍明显低于archived Full972 seed32，Point也只有102/108，11项最终晋级门中5项失败。因此分支停止，`candidate_authorized_for_test=false`，没有访问test270，也没有更新Leaderboard。

通俗地说：把同一个学生最后六次答题习惯平均，确实连续三次都比只拿最后一次稍稳；但第三个学生本身的随机训练轨迹不够强，这点小修补救不回与历史强学生之间的更大差距。

## 问题、直觉与为什么选择

开始时，reused internal test270最高Scan是GCSAM`0.752003`，但Point只有252/270；Full972 voxel-logit knowledge distillation（逐体素logit知识蒸馏，简称voxel-KD）Scan只低`0.000244`，却有262/270 Point。Fresh seed之间inner Scan又可从`0.732792`变化到`0.746412`，显示随机训练轨迹稳定性仍是瓶颈。

本实验不改模型、loss、teacher或数据，只平均同一seed的epoch163–168参数。Cross-seed parameter soup失败的是不同权重盆地；旧指数移动平均使用Full756旧轨迹和step级长记忆；GCSAM改变优化更新而不做同轨迹平均，所以这些负结果没有预先否定本实验。

## 流程与唯一变化

输入T1ce经既有image-only brain crop与true-128³ cache后进入4,700,897参数plain SegResNet。训练仍使用真实WT标签的Dice加0.1交叉熵，以及固定`0.75 history168 + 0.25 EMA168`teacher的`0.3×T²`逐体素Kullback–Leibler divergence（KL散度），温度`T=2`。

唯一变化是最终权重：

- Control：同一run的raw epoch168。
- Candidate：同一run的epoch163、164、165、166、167、168六份参数以CPU float64精确等权平均，再转回原dtype。

数据、每个run的seed/初始state、972 requested/968 consumed/121步、batch8、native-volume sampling、增强、AdamW、晚余弦schedule、阈值0、无测试时增强、无Segment Anything Model、无后处理和1×推理全部固定。平均权重从不反馈到训练。

## Stage A：两个fresh train-only inner seed

新split为864 fit/393 subjects与108 inner/51 subjects，case和subject重叠均为0。两个fit864 seed均训练到epoch168，所有六次捕获、reload/deploy、finite、memory、time和access门通过。

| Fresh train-only inner108 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| seed61 raw168 | `0.744803721` | `0.731433000` | 103/108 |
| **seed61 LAWA6** | **`0.747122826`** | **`0.733388320`** | **106/108** |
| LAWA6 − raw | `+0.002319105` | `+0.001955320` | `+3` |
| seed62 raw168 | `0.737862323` | `0.726117012` | 106/108 |
| **seed62 LAWA6** | **`0.739130026`** | **`0.728889501`** | **107/108** |
| LAWA6 − raw | `+0.001267702` | `+0.002772489` | `+1` |

- seed61 Scan/Subject 95% CI：`[+0.000345,+0.004336]` / `[-0.000520,+0.004880]`；69/108 scans改善、38变差、1持平；29/51 subjects改善、21变差、1持平。
- seed62 Scan/Subject 95% CI：`[-0.001534,+0.003764]` / `[-0.000239,+0.005689]`；67/108 scans改善、40变差、1持平；35/51 subjects改善、15变差、1持平。
- 四个CI下界都高于冻结`-0.003`安全门，两个seed的Scan/Subject点估计均严格提高且Point不降，所以Stage A 10/10门通过，授权Stage B。

## Stage B：fresh Full972 final与fixed validation

第三个fresh seed20261063在full972 scans/444 subjects上从随机初始化训练到epoch168。所有训练工程门通过后，才一次性访问fixed validation108/50 subjects。

| Fixed validation108 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| fresh seed63 raw168 | `0.732455624` | `0.725793200` | 102/108 |
| **fresh seed63 LAWA6** | **`0.732919459`** | **`0.726369588`** | **102/108** |
| LAWA6 − raw | `+0.000463835` | `+0.000576388` | `0` |
| archived Full972 seed32 | `0.740012535` | `0.731266217` | 103/108 |
| LAWA6 − archived | `-0.007093076` | `-0.004896629` | `-1` |

配对分布与置信区间：

- 相对matched raw：Scan 95% CI`[-0.001387,+0.002170]`，64/108 scans改善、44变差；Subject CI`[-0.001494,+0.002629]`，29/50 subjects改善、21变差。两项CI安全门通过。
- 相对archived seed32：Scan CI`[-0.015439,+0.000068]`，50/108 scans改善、58变差；Subject CI`[-0.015411,+0.004016]`，19/50 subjects改善、31变差。两项CI安全门失败。

最终11门通过6项：超过matched raw的Scan/Subject、Point不低于raw、两项matched CI安全和工程门。失败5项：未超过archived Scan/Subject、Point未达103、相对archived两项CI安全门失败。因此不访问test。

## 直觉是否得到支持

**局部机制得到支持，完整超榜目标不支持。** 三个独立fresh seed的LAWA6相对各自raw都提高Scan与Subject；Stage A Point分别增加3和1，Stage B Point持平。这不是单个seed偶然正号，说明当前固定六轮平均能轻量平滑同轨迹末期抖动。

但收益规模只有约`+0.00046`到`+0.00277`，第三个fresh final与archived强seed仍差`0.00709/0.00490`，表明seed间表示学习差异远大于最后六轮的盆地内抖动。该方法不能解决“偶然强seed不可稳定复制”的主要瓶颈。

当前只否定固定Full972 T1ce/WT、AdamW晚余弦、epoch163–168、uniform`k=6`这一完整晋级主张；不证明所有同轨迹平均都不可能有效。但按冻结规则不在已揭示inner/validation上改`k`、窗口、系数、epoch、seed或阈值。

## 成本与部署

- Stage A两个fit864 run：`12,207.23s / 12,147.83s`，各约3.39/3.37h，双卡并行墙钟约3.39h。
- Stage B Full972 run：`12,218.44s`，约3.39h。
- Formal总GPU时间约10.15 GPU-hours；实际串行阶段墙钟约6.78h。
- 每个run peak allocated`21,796,646,912` bytes，约20.30GiB。
- Workspace约181M，复用现有cache，没有创建大cache或删除材料。
- 最终candidate仍4.70M参数、1×locator forward；不增加推理延迟、teacher、TTA、SAM或postprocessing。

## Evidence与访问

- IDEA SHA-256 `b2cfc6b6b01624d0efa564c8f019b15f65b6892bf96fd76c1ec6331d8dc3092d`。
- PREREG SHA-256 `cd72ab82bdadf5cd574a40f353cdf10218d4eace38ebe97ada3edc3f9a66cfc9`。
- Stage A raw inner SHA-256 `09cde688b29f8b98562acce5788c473475ad89781b0c4acd7c0a08c09fd51d6d`。
- Stage A decision SHA-256 `2437a641c28c218878e5df45c784b2b3154718416d0dd920a5ca10b10ad59001`。
- Stage B train report SHA-256 `7bf7c7e1dcc664067bc5db4cb68d565e12f6fba7cac858a0ad2fcf22983a7bf3`。
- Stage B validation pre-access receipt SHA-256 `d4b068b1cedca1ce604807132a519575391bb58d4733f428e5bed1ed7df206ad`。
- Stage B raw validation SHA-256 `c1be148cbb57f6e144593f47b0074bb1a85b4fd99acc6beb309719654a14df6d`。
- Selected LAWA6 checkpoint SHA-256 `aac84cf2970159a50c0b5b442cff1ff54cb7557c4a01d97b1d86d4e25ff160f3`；model-state SHA-256 `09fa2af2fcbfb8259f39e60ea393e9e97b09821ecd32910f0f1458a40c478847`。
- Fresh train-only inner与fixed validation108已访问；outer、confirmation和test270均未访问。

## 下一决定

关闭当前LAWA6分支，不调`k`、窗口、uniform coefficient、epoch、seed、loss、threshold或postprocessing，不进test。后续内部研究不能继续做同轨迹窗口/权重搜索，也不能回到已关闭的多seed融合；真正提高证据等级仍优先需要独立外部cohort。若继续内部development，应转向尚未解决的数据表示/训练信息瓶颈并建立新的fresh证据。

Evidence links: [七问](IDEA.md)、[预注册](PREREG.md)、[Stage A决定](STAGE_A_DECISION.json)、[raw Stage A inner](../../../workspace/BraTS/sammed3d_t1ce_wt/full972_lawa6_stability_20260902/STAGE_A_INNER.json)、[raw Stage B validation](../../../workspace/BraTS/sammed3d_t1ce_wt/full972_lawa6_stability_20260902/STAGE_B_VALIDATION.json)、[Stage B checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/full972_lawa6_stability_20260902/formal_stageb_full972_seed20261063/lawa6_stageb_full972_seed20261063_epoch168.pth)。
