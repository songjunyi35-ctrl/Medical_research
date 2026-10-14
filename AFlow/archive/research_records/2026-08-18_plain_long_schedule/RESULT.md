# Plain locator late-cosine 结果（2026-08-18）

## 结论

100 epoch 对本轮新鲜 subject-clean inner 轨迹确实偏短。固定前100轮 `lr=2e-4`，随后用
预注册 cosine continuation 后，inner 最佳从 epoch95 的 scan/subject
`0.723604/0.719629` 提高到 epoch168 的 `0.746335/0.744497`。scan/subject 配对差为
`+0.022731/+0.024869`，subject 95% bootstrap CI 为
`[+0.014792,+0.038213]`，五个 inner gate 全部通过。

按协议锁定 epoch168 后，在原756-case train role 上从头训练，不读取任何 validation，
再对 current outer selector 只评估一次。新的固定 checkpoint 达到 scan/subject
**`0.731008/0.714314`**、automatic point hit **`0.925926`**；相对冻结的 epoch99
baseline 为 **`+0.010589/+0.009697/-0.009259`**。三项预注册 outer 主门槛全部通过，
因此 epoch168 checkpoint 升级为新的 development candidate。

完成上述模型选择与outer决策后，用户明确授权把已固定的epoch168 checkpoint在270-case
internal test上评估一次。test不参与epoch、阈值或checkpoint选择，结果也不回流调参。
epoch168取得scan/subject/point hit **`0.741204/0.723179/0.929630`**，相对epoch99为
**`+0.015384/+0.013767/+0.029630`**；scan和subject配对95% CI下界均大于0。

## 1. 冻结协议

- 模型不变：plain `AutomaticPromptLocator` / SegResNet，4,700,897参数；
- 输入/目标不变：T1ce → WT `{1,2,3}`，image-only brain crop `128^3`；
- loss、sampling、augmentation、batch=8、AdamW和raw logit threshold `>0`均不变；
- epoch 1–100：`2e-4`；epoch 101–200：cosine `2e-4 → 1e-5`；
- inner完整训练200轮，只比较同轨迹 pre100/post100 两个窗口；
- outer阶段只能把inner选出的单一epoch用于一次fixed-epoch重训和一次selector评估；
- confirmation/test、CAPG、SAM、TTA、connected components和灰度阈值均不进入本实验。

新inner只从2026-08-17实验的648 fit cases重新切分，昨天已揭示的108 inner-tune完全
排除。新角色为540 fit / 249 subjects与108 fresh tune / 52 subjects，case和subject均
零重叠。预注册 SHA-256：
`035cda371ee8ac9d57bf0558d7130534579fad539a72903a486cb1e4a123bc48`。

## 2. Fresh-inner机制结果

| window | epoch | lr | scan Dice | subject Dice | point hit |
|---|---:|---:|---:|---:|---:|
| pre100 best | 95 | `2.0e-4` | `0.723604130` | `0.719628735` | `0.935185` |
| post100 best | 168 | `5.491358e-5` | **`0.746334987`** | **`0.744497488`** | `0.925926` |
| post − pre | — | — | **`+0.022730857`** | **`+0.024868753`** | `-0.009259` |

subject配对bootstrap为10,000次、seed `20260824`；95% CI
`[+0.014791766,+0.038213440]`。两份checkpoint均独立重载完整108例，逐病例Dice最大
绝对差为0。`INNER_DECISION.json` SHA-256：
`6308c5e3807e513cc0eb3aebf0c4718e29c7e04e5748b314da6f3b28a7b227b4`。

这个比较证明“预注册的额外epoch + late cosine组合”优于同一训练轨迹的前100轮；它
不能把收益进一步拆成纯epoch效应与纯scheduler效应。

## 3. 全756例固定重训与outer结果

inner通过后，使用原756 train cases / 346 subjects、seed `20260819`从头训练到
exactly epoch168。训练脚本没有validation DataLoader，训练前后
`validation/selector/confirmation/test access`均为false。训练耗时`3448.77 s`，峰值
GPU allocated `21,549,951,488 B`。

| current outer selector | epoch99 baseline | epoch168 candidate | candidate − baseline |
|---|---:|---:|---:|
| scan-macro Dice | `0.720419292` | **`0.731007992`** | **`+0.010588700`** |
| subject-balanced Dice | `0.704616933` | **`0.714314407`** | **`+0.009697474`** |
| automatic point hit | `0.935185` | `0.925926` | `-0.009259` |

outer成功线为scan和subject各至少`+0.003`、point hit下降不超过`0.02`；全部通过。
描述性subject paired bootstrap 95% CI为`[-0.001312,+0.020934]`，跨0且不是预注册
outer gate。因此正确表述是“预注册development指标突破”，不是统计显著提升。

## 4. 为什么会提高

本轮没有增加模型容量，也没有改阈值或增加推理后处理；推理参数量和成本不变。可直接
支持的解释是：旧100轮上限在新鲜inner轨迹中截断了后续学习，低学习率收尾让同一个
dense locator获得更好的分割解。outer上的post-hoc分布也更像广泛改善，而不是少数
异常病例或中心点投机：

- 108 scans中80提高、26下降、2持平，病例Dice差中位数`+0.010743`；
- 45 subjects中33提高、12下降，subject差中位数`+0.008506`；
- baseline Dice与病例增益相关系数仅`-0.0554`，收益不只来自原本最差病例；
- point transition为97 hit→hit、4 hit→miss、3 miss→hit、4 miss→miss，净少1个hit。

这些是冻结输出上的解释性统计，不用于选择模型。没有边界/病灶尺度消融，不能进一步
声称收益来自某一种解剖区域。灰度残差、CAPG中心头和旧冻结SAM后处理的负结论不变。


## 5. 固定epoch168的test270结果

test前冻结checkpoint SHA、outer决策、raw logit `>0`、无TTA/后处理/SAM和唯一输出路径。
270 scans / 119 subjects与756 train和108 selector在case/subject上零重叠。

| reused internal test270 | epoch99 | epoch168 | 配对差 | 配对95% CI |
|---|---:|---:|---:|---:|
| scan-macro Dice | `0.725820068` | **`0.741203949`** | **`+0.015383882`** | `[+0.009186,+0.021574]` |
| subject-balanced Dice | `0.709412503` | **`0.723179486`** | **`+0.013766983`** | `[+0.004123,+0.022935]` |
| automatic point hit | `0.900000` | **`0.929630`** | **`+0.029630`** | — |

epoch168自身scan/subject bootstrap 95% CI为`[0.722216,0.758635]`和
`[0.695220,0.748559]`。270 scans中205提高、64下降、1持平；119 subjects中87提高、
32下降。point transition为231 hit→hit、12 hit→miss、20 miss→hit、7 miss→miss。

因此epoch168是本项目当前复用270 internal test上的新历史最高点估计，且配对scan与
subject CI均为正。由于该集合此前已被多个arm揭示，仍不能称pristine blind test、外部
确认或公开BraTS SOTA；但它是对训练/验证阶段已固定方法的一次合法最终测评。

## 6. 推荐checkpoint与证据身份

- 新的development candidate：
  `workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth`
- checkpoint SHA-256：
  `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`
- 该checkpoint现在同时通过fresh inner、outer selector和固定test270测评；适合下一次
  真正外部cohort评估。
- 不得再用已揭示的selector、confirmation或270 test选择epoch、阈值、TTA、SAM decoder
  或后处理；本次test结果不触发继续调参。

## 7. 核心证据

| artifact | SHA-256 |
|---|---|
| `PREREG.md` | `035cda371ee8ac9d57bf0558d7130534579fad539a72903a486cb1e4a123bc48` |
| inner `TRAIN_DONE.json` | `cf7e11b799413f03f2350f07559be2d71063b2fa30e3bfec0289c3a238b61774` |
| `pre100_best.pth` | `d83a25f2da88ffd9274b4ffbfaf2593a1f2e127739c37b2432fb8ea2c8ccf911` |
| `post100_best.pth` | `cdbbe9d7c4c5a7e2ae8c2018d0e3310b4ea0b522a1c126f99ffb965e83330662` |
| `INNER_DECISION.json` | `6308c5e3807e513cc0eb3aebf0c4718e29c7e04e5748b314da6f3b28a7b227b4` |
| outer pre-access freeze | `cf399a469d1546a5ff5f262f34d881be4f4bc3982bb8a0e4fe7e68b14fd3fdc8` |
| fixed-epoch `TRAIN_DONE.json` | `a027423080f8b44d39cb2b538e50a4685393b4bc1c74dd46b601aa4682885fd3` |
| fixed epoch168 checkpoint | `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1` |
| one-shot outer report | `310ea3cc20343695b23fc1841df66d7d627175ac5082836c5e82144a353f60ff` |
| `OUTER_DECISION.json` | `f8f876e6e04b31300c578c9bd2f235cd1b74948f0139f9810a5c4d67b4711ca7` |
| test270 pre-access freeze | `30251c67669c03d4eaebcaeb445dac1d45c193392717f26a0147e8e97e79d750` |
| epoch168 test start receipt | `dc358933608a8f4a01a60caa3b773c3d3e0d223db7f04e145c07c0e8dcc30d2e` |
| epoch168 `FINAL_TEST270_EPOCH168.json` | `91433c4b6b88b233d827e41b0d038f459c25ef41dec224c84e154e90f370bb8c` |
| epoch168-vs-epoch99 paired report | `62792337648b39d6080c88c9c67be915741a0c8176b1a6353c9833577d7e7a9b` |

全仓回归：`273 passed, 24 warnings`。warning均为既有SciPy deprecated namespace，
没有测试失败。
