# WAW152事后误差审计边界

状态：正式结果已经揭示后的描述性分析；不是预注册性能检验，不新增模型forward，不授权选择或调参。

## 问题与直觉

WAW152上GCSAM−Full identity Dice=`+0.010223`但区间跨0，Flip8−identity=`+0.010116`且区间全正。Dice是“预测和真值重叠有多大”的综合分数；它可能来自少漏画（recall提高）、少多画（precision提高），也可能只集中在某种肿瘤大小或WHO grade。像总分相同的两位学生，必须拆开看“漏题”和“错题”才知道机制。

## 冻结只读做法

- 只读`RAW_WAW152.json`保存的intersection/predicted/target计数、冻结cohort的WHO grade和WAW/train972 cache index；不读取logit/mask、图像或checkpoint。
- 对三模型计算逐例precision、recall、prediction/target体积比；对GCSAM−Full和Flip8−Full给事后10,000次病例bootstrap区间，seed `20261453`及偏移固定。
- 按真实target体积排序后固定四个各38例quartile；按已发布grade2/3/4分组；把WAW target和预测体积映射到train972经验分位。
- 画出整体precision/recall、体积四分位差、grade差和训练分位。所有分组标`post-hoc descriptive`，不做多重比较显著性宣称。

## 成功、失败和禁用

没有promotion gate。输出回答“差值主要伴随precision还是recall、在哪些已见亚组更大”；无论模式如何，都不能把target体积、grade或scanner变成router、阈值、checkpoint选择器，也不能在WAW/旧cohort复测。不同阶段、label语义、发布者预处理和scanner共同变化，相关性不等于因果。

## 冻结输入

- WAW raw SHA-256 `ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91`
- WAW cohort SHA-256 `25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a`
- WAW cache index SHA-256 `a3831454785715ba48f845115571f99b8adfe61011d29fadaa0b91a4929adc7e`
- train972 cache index SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`
