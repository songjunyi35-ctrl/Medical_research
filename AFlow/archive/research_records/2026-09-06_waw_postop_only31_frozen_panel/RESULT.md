# WAW术后only-patient31冻结面板：最终结果

状态：一次性public early-postoperative external完成并关闭；与preop152患者零重叠，非hidden blind，不调参/重跑。

## 初学者结论

从WAW final postop中先排除所有已经参加preop152考试的患者，再对每名新患者只取最早且日期明确的一张，得到31人、术后0–22天。三种方案Dice约0.59–0.60，差距很小且配对区间都跨0；Full固定八镜像Dice点估计最高，GCSAM是1× Dice点估计最高，Full identity与Flip8的Point同为15/31。没有确定赢家。

| 方法 | Scan=Subject Dice（95% CI） | Point hit | forwards |
|---|---:|---:|---:|
| Full972 KD168固定Flip8 | **`0.598959388`** `[0.532228,0.662012]` | **15/31 (`0.483871`)** | `8×` |
| GCSAM184 identity | `0.593304725` `[0.527500,0.654293]` | 14/31 (`0.451613`) | `1×` |
| Full972 KD168 identity | `0.591647416` `[0.522240,0.654848]` | **15/31 (`0.483871`)** | `1×` |

## 配对解释

- Flip8−Full identity=`+0.007311972`，paired 95% CI=`[-0.001925,+0.016626]`，21改善/10变差/0 tie，Point差0。按冻结门是`positive_but_uncertain`，不支持确定外部收益。
- GCSAM−Full identity=`+0.001657309`，CI=`[-0.012498,+0.016205]`，17/14/0，Point−1，同样`positive_but_uncertain`。

这给GCSAM增加第六个cohort但仍只有Africa95的配对区间全正。Flip8第五个cohort仍是正点估计，但本次区间跨0，和Africa一样不确定。N=31、早期术后变化大，不能把点估计排序变成显著性。

## 只改变、成本与访问

唯一新增cohort：final160 postop exams中先排57名preop-overlap患者；35名post-only里排4名日期全空，再对31人固定最早numeric day，排17张后续scan。29/31为WHO4，WT raw labels1/2/3、排resection bed4。模型/checkpoint、N4/crop/z-score/128³、阈值0、view、无SAM/后处理和bootstrap不变。

- 训练0；官方postop zip3,321,955,416 bytes，选中31 T1c 153,907,032 bytes；cache约310MiB、141.91秒。
- 正式279 forwards、5.76秒、峰值5,457,695,744 bytes（约5.08GiB）。Flip8仍8×，identity1×。
- 只访问WAW-postop31；preop WAW/internal/fixed validation/outer/confirmation/UCSD/Africa/CPTAC均未访问。
- 同Warsaw机构但新患者、相对BraTS七站institution-disjoint/no patient bridge；public非盲。Day0–22是early post-resection，不代表长期复发。Publisher预处理后项目再N4/128³，训练源parity caveat保留。

## 事后错误边界：最高点被切除腔吸引

用保存的argmax索引和原始多类label离线重建128³ WT/RC，并先要求WT与正式cache 31/31 byte-exact。Full identity/Flip8的Point类别均为WT15、resection cavity13、普通背景3；GCSAM为WT14、RC14、普通背景3。也就是Full的16个miss中13个、GCSAM的17个miss中14个落在GT切除腔，解释了早期术后Point约一半，而Dice仍约0.59。

Full/Flip8/GCSAM平均precision=`0.567/0.587/0.570`，recall=`0.658/0.654/0.665`，prediction/target ratio=`1.251/1.217/1.311`。Flip8仍主要减少多画，GCSAM稍增recall；都不是冻结promotion检验。GT RC在部署不可用，严禁据此做RC避让后处理或重测；未来若自动预测RC，必须新训练/新证据。[事后审计](POSTHOC_ERRORS.json)、[图](POSTHOC_ERRORS.png)。

按metadata事后分组：day0–1的24例中，Full/Flip Point为WT9/RC12/背景3；day>1的7例为WT6/RC1/背景0。GCSAM对应9/12/3与5/2/0。最早期RC吸引更集中，但后组仅7例，时间与患者/手术组成共变；禁止按day选模型或声称恢复轨迹。Dice day0–1/`>1`也没有同步大幅反转（Full`.598/.568`、Flip`.601/.591`、GCSAM`.600/.569`），进一步说明Point是独立失败轴。

一手资料支持“RC/CSF/坏死核心是已知术后混淆源”且BraTS把recent/chronic RC单列label4；它解释现象但不提供当前模型可直接套用的修复。显式RC head、CSF segmentation或surgery metadata都属于新模型/输入，必须新训练和新证据。[文献边界](RC_LITERATURE_CONTEXT.md)。

当前主系统是direct locator：mask直接由`logit>0`输出，无SAM，Point只是同一logit最高位置的诊断。因此RC argmax不会回灌或改变本次Dice；它警告的是未来若把最高点当SAM/交互prompt，早期术后可能落在错误结构。不能把“Point低”误写成当前mask由错误prompt产生。

与已归档preop汇总对照，postop WT mean/median仅为preop `1.017×/1.071×`，cache std/nonzero-fraction median差仅`+0.000334/+0.00249`。粗体积、尺度与脑区支持几乎相同却Point骤降，不能用“术后WT更大”解释；RC与grade/患者组成更相容，但两组患者不同，仍非因果。[域addendum](../../reviews/2026-09-06_fresh_evidence_capacity/WAW_POSTOP31_DOMAIN_ADDENDUM.json)。

进一步只保留WHO grade4仍看到大差：pre74 vs post29的Full/Flip/GCSAM Dice=`.730/.745/.737` vs `.586/.594/.586`，Point=`74/74,74/74,72/74` vs `14/29,14/29,13/29`；post target median反而小13,546.5 voxels。故grade组成或更大WT都不是充分解释。两组仍是不同患者，post28/29含RC、pre0/74，不能把差值`−0.144至−0.150`称手术因果。[Grade4描述](POSTHOC_GRADE4_PRE_POST.json)。

## 证据

- [七问](IDEA.md)、[预注册](PREREG.md)
- [raw](RAW_WAW_POSTOP31.json) SHA `72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98`
- [模型前收据](MODEL_PREACCESS.json) SHA `e8e13d35491672a641f4f7879a3f627a45438368fc04febcf341040c54a3acad`
- [机器结果](RESULT.json) SHA `6176b2c3562596a660238e104421095a6a1a28868177846efd4fd88e0c42ed9f`
- [终态验证](VERIFICATION_RECEIPT.json) SHA `b4803a989ec3b3450918dc13e60b227cf8b59701557b68bea5cf2e153b32ea76`
- [事后错误审计](POSTHOC_ERRORS.json) SHA `a49b2985eaf8635b9a3bc1b8e8bee78eb137cc87a94bc599620a3fa0ddd3cc0c`
- [事后图](POSTHOC_ERRORS.png) SHA `03723810b2e8ec1ad952612273af760ed6623cdcc00644df6ea866178ba2ce09`
- [Grade4 pre/post描述](POSTHOC_GRADE4_PRE_POST.json) SHA `663d79aa5208f545ecc64bbbf43627547f6b4adb393f5334ec85644dfcee109b`
- [来源](../../reviews/2026-09-06_waw_postop_only31_source_audit/RESULT.md)
- [Full checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth) SHA `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`
- [GCSAM checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth) SHA `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`

WAW-postop31关闭：不得换后续scan、空日期患者、阈值、view、模型或预处理重测。
