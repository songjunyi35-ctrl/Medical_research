# Full972 width24完整轨迹结果

## 结论

最终validation失败，停止且不访问test。把SegResNet从4.70M参数加宽到10.57M，在train-only inner上看起来更好，但换新seed、用完整972例重训后没有复现：width24的Scan、Subject和Point都低于matched width16。

通俗地说，这像一次模拟考中“大号模型”领先，但正式同卷考试里反而少做对了一些题。说明inner收益不稳定，当前数据和训练配方下，模型容量不是主要瓶颈。

## Fixed validation108 / 50 subjects

| 方法 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| Archived Full972 width16 seed32 epoch168 | `0.740012535` | `0.731266217` | 103/108 |
| Matched Full972 width16 seed20260961 epoch166 | `0.739335220` | `0.731320350` | 105/108 |
| Full972 width24 seed20260961 epoch166 | `0.736463710` | `0.727794977` | 104/108 |

Width24相对matched width16：Scan `-0.002871510`，95% CI `[-0.014258090,+0.005947492]`；Subject `-0.003525373`，95% CI `[-0.013329821,+0.005515228]`；Point `-1/108`。108个scan中54个改善、54个变差；50个subject中24个改善、26个变差。

Width24相对archived seed32：Scan `-0.003548825`，95% CI `[-0.013144105,+0.005200474]`；Subject `-0.003471240`，95% CI `[-0.013131386,+0.005784568]`；Point `+1/108`。

## 方法与成本

两臂均用Full972、seed20260961、968有效draw/121步、同一history/EMA teacher、DiceCE+voxel-KL、200轮schedule并固定停在inner选择的epoch166。唯一变化是`init_filters=16→24`，参数`4,700,897→10,574,161`；阈值0、无TTA/SAM/postprocess，均为单模型1×forward。

Width16训练约3.56h；width24约4.86h。Width24参数约2.25倍，训练更慢，但没有换来validation收益。

## Promotion与访问

Width24超过matched width16和archived seed32的四个点估计门均失败；四个paired CI安全门也失败。Point≥101和全部工程门通过，但不足以晋级，因此`candidate_authorized_for_test=false`。

Fixed validation108已访问；outer selector、confirmation和test270均未访问。不会用本结果回调宽度、epoch、阈值或后处理。

Raw validation报告SHA `d2152ea897740fdfbbf1e1665e5f60e2ea1819a392cb00ccce231674b8eb5608`；pre-access receipt SHA `caf78abd3aaf4031b00ea50d7a2887bc103a5bdd3a102b3a1ecf89e910e635e9`。

## 下一决定

关闭当前width24方向，保留4.70M width16作为容量配置。下一项科研应换机制并重新预注册，不能在本validation上搜索width20/32或改epoch。
