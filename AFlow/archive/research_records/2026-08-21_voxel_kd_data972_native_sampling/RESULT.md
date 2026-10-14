# 972例原生体积感知voxel-KD采样结果

## 结论

原生体积采样完成了它最直接的目标：把上一次972数据扩容的Point从99/108恢复到101/108，同时selected candidate200仍超过新same-epoch native756 control的Scan/Subject。但它没有突破历史KD200 Scan，差`0.000324`，因此按冻结协议只失败`beats_kd200_scan`一门，不访问test270。

这条结果进一步支持“数据多样性有用”，也说明Point下降确实部分来自resize体积采样口径；但当前fixed-compute 972 pool还不足以提高overall Scan超过KD200。

## 数据处理与实现边界

- 原生体积定义为canonical RAS原生1mm网格中、冻结image-only brain crop内的WT `{1,2,3}`体素数。
- 972 train-only工件全部体积为正；重叠756/756与历史formal index逐例exact。
- native756完整float64权重张量与历史KD权重SHA同为`757e463d...2124`，max-abs差0；native972权重SHA为`81d3fa6a...48ec`。
- 正式前exact门暴露了full-FOV/crop定义、LAS/RAS坐标与Python布尔拼写三个问题；都在工件输出、模型训练和validation前停止并透明记录。没有根据任何结果指标修正方法。
- 虽然权重和缓存exact，native756训练指标没有bit-exact复现历史KD轨迹，因此它只被称为新same-runner matched control；历史KD200仍是独立第二基准。

## 唯一变化与固定项

- 两臂唯一差异：可抽样病例池756→972。
- 两臂统一使用原生crop WT体积权重、同一旧972 cache、每轮756 draws/94 steps。
- history168 student、history/EMA teacher、DiceCE、`0.3` voxel-KL、`T=2`、seed20260846、batch8、32轮late-cosine、shift8、AdamW历史默认、clip5、threshold0和1×部署均固定。

## Fixed validation108

| 方法 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| **native972 candidate epoch200** | **`0.734059862`** | **`0.724702811`** | **`101/108`** |
| native756 control epoch200 | `0.733370942` | `0.724036493` | `102/108` |
| archived KD200 | `0.734384308` | `0.724434143` | `101/108` |

相对same-epoch control：

- Scan `+0.000688919`，95% CI `[-0.001192860,+0.002624611]`。
- Subject `+0.000666318`，95% CI `[-0.001717258,+0.003066053]`。
- Point `-1/108`；49/108 scans、22/50 subjects改善。

相对历史KD200：

- Scan `-0.000324446`，95% CI `[-0.002921337,+0.002036791]`。
- Subject `+0.000268667`，95% CI `[-0.002422631,+0.002868334]`。
- Point持平101/108；57/108 scans、23/50 subjects改善。

四个candidate Scan为176 `0.733911`、184 `0.734009`、192 `0.733440`、200 `0.734060`；只按冻结的Scan→Subject→早epoch选择200。Epoch184 Subject更高`0.725207`，但不能事后按Subject改选。

## Promotion

通过：超过same-epoch control Scan/Subject；超过KD200 Subject；Point≥101；四项CI安全门；native-volume/weight exact、11项测试、V2 smoke、formal、8 checkpoint reload、资源和access门。

失败：Scan没有严格超过KD200。

因此`candidate_authorized_for_test=false`。不能因为只差`0.000324`就改选epoch184、增加draw、延长训练、改sampling clip或访问test。

## 成本

- 原生体积工件433KB，无新大cache。
- 两臂formal约1937.2/1938.9秒，两卡并行墙钟约32.3分钟。
- 每卡peak allocated约20.29GiB。
- Workspace约217M；部署仍4,700,897参数、1×forward。

## Evidence与访问

- Raw validation SHA-256 `963efa941b1e128243525fad0ac30563d362926ec4264b4b7bd9c15e4894ad37`。
- Selected checkpoint SHA-256 `b57693c6af1cded8de4c4e576f462ff203c942ec7cd7a78a3711b70394470c4b`。
- Native artifact SHA-256 `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`。
- Control/Candidate TRAIN_DONE SHA-256 `32772dceafa1f542b4a239dff74efc19b86c6e02673d41edef37cb8ca8f7f122` / `3c876f8b0a882e6f20a229d6fdacbbad01ce6984cface77d6ae83dac6236af26`。

本方法的新fixed validation已访问；outer/confirmation已回收为train角色；test270未访问。

## 下一决定

当前native-volume fixed-compute协议停止。数据多样性连续两次超过matched control，但始终没有超过KD200 Scan。若继续数据轴，下一条不应再调sampling公式，而应重新评估固定计算量假设：972病例每轮只抽756次使每病例平均更新减少。任何`972 draws/epoch`实验都会同时增加约28.6%计算量，必须作为新的“数据＋计算预算”目标重新预注册，不能由本validation直接启动test。

