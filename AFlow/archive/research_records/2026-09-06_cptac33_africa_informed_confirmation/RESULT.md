# CPTAC33 Africa-informed冻结checkpoint确认结果

状态：✅ 一次性公开Training33外部确认完成并关闭；主假设弱/不确定支持，Flip8 secondary方向通过。

## 初学者结论

Africa95第一次显示GCSAM184比Full972 KD168更会圈完整肿瘤，Dice高`0.0686`。这次到另一套CPTAC术前公开人工标签复考，GCSAM仍略高，但只高`0.00195`，置信区间跨0、33例中16胜17负。因此Africa的大优势没有强复现，只能称方向相同但不确定。

Full972固定8镜像平均则从identity的`0.841331`升到`0.844577`，差`+0.003246`且配对区间整体为正，23胜10负，Point都是33/33。这给Flip8提供一份新队列的Dice方向支持；它仍要8倍推理，不能由此搜索视图或权重。

## 唯一变化与固定项

唯一新增CPTAC-GBM/BraTS2021 public Training33。Validation6因为官方无seg在payload前排除，hidden test未用。面板与Africa完全相同：Full KD168 identity/固定Flip8、GCSAM184 identity，all-positive术前WT、canonical RAS、N4 shrink2、image-only crop、z-score、128³、阈值0、无SAM/后处理/模型ensemble、10,000 bootstrap均固定。

Africa结果在CPTAC payload前明确形成主假设；CPTAC结果不用于改变任何设置。两个checkpoint仍非optimizer-only配对，外部差值只比较预测，不做训练因果归因。

## 工程证据

- 官方mapping：CPTAC39=Training33+Validation6；Validation6全部0 seg，39方案原样失败。
- 官方`.sums`49,850,968 bytes/407,245 MD5 rows，选中66/66覆盖；双源132文件/156,027,148 bytes在完成时重hash，官方MD5/size/双SHA全exact。
- 首例240×240×155、1mm、RAS、affine差0、raw `{0,1,2,4}`；全33 raw union同样`{0,1,2,4}`，WT全非空。
- Cache耗时288.054秒，首例probe image/label byte exact；image/label SHA=`47cb710c…/e4202ba2…`。
- 首次formal因preaccess字段路径写错在checkpoint/model前KeyError，0 forward；V2收据修正嵌套字段后才唯一一次运行。

## 绝对结果

| 方法 | Scan=Subject Dice（95% CI） | Point | 成本 |
|---|---:|---:|---:|
| Full972 KD168 Flip8 | **`0.844577117`** `[0.813290,0.870795]` | 33/33 | `8×` |
| GCSAM184 identity | `0.843285080` `[0.812977,0.868437]` | 33/33 | `1×` |
| Full972 KD168 identity | `0.841331464` `[0.809100,0.868613]` | 33/33 | `1×` |

每人1 scan，所以Scan=Subject且bootstrap区间相同。

## 冻结比较与Africa假设

- GCSAM−Full identity：Dice=`+0.001953616`，95% CI=`[-0.006645827,+0.012383793]`，16/17/0 improve/worse/tie，Point差0。按预注册为`weak_uncertain_support`；Africa的强差没有独立复现。
- Full Flip8−identity：Dice=`+0.003245653`，95% CI=`[+0.000827765,+0.005854744]`，23/10/0，Point差0。Direction及CI支持，但只适用于当前固定8视图。

CPTAC target平均76,873 voxels；Full identity/Flip8/GCSAM平均预测70,702/70,959/74,547，预测/真实比均值约`0.949/0.951/0.999`。它不像Africa的`0.50–0.61`系统漏分，解释了Africa GCSAM coverage优势为何不能原幅复现；cohort差异仍不是单因实验。

## 成本与访问

- 训练0；source双份约149MiB，`.sums`47.5MiB，cache约330MiB，均有hash。
- Cache288.1秒；正式297 forwards 6.549秒，峰值5,457,695,744 bytes（5.08GiB）。
- Full/GCSAM identity部署1×，Full Flip8 8×。
- CPTAC33访问=true；internal test270、UCSD、Africa95新访问、fixed validation、outer、confirmation均false。
- Public Training external-to-project、not hidden blind、source-disjoint/no patient bridge；checkpoint训练源官方Synapse parity未解决。
- Cohort关闭：不调模型/阈值/视图/病例，不重跑。

## 证据

- [七问](IDEA.md)、[预注册](PREREGISTRATION.json)
- [来源结果](../../reviews/2026-09-06_cptac_brats2021_source_audit/RESULT.md)
- [官方sums收据](SUMS_RECEIPT.json)，SHA `9d73c686c565aa238dfa65648c041557f9d434394aa7e4e558eb38471a1b381b`
- [双源](SOURCE_DOWNLOAD.json)，SHA `8c23620f6800c4f6d885e346b0b7ccef35149f0dba3fba9b868af1dfed2e8bb8`
- [V2模型前收据](MODEL_PREACCESS_V2.json)，SHA `117cb7c38000107f44823f204d171bf13a4ce8a2bc84e96487b670abb1900f09`
- [raw](RAW_CPTAC33.json)，SHA `4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd`
- [started](RAW_CPTAC33_STARTED.json)，SHA `58401d39fe82cbf06cf4c42392fe3b0ee85732508465428eff9f8d3146cd1ca0`
- [cache index](CACHE_INDEX.json)，SHA `9eeebdf66779496259b3a909ae8262dc380f3eaf73544059383b0d6c003cd140`
- [终态验证](VERIFICATION_RECEIPT.json)，SHA `52188762aed8d81f7a5a75dbce79fe06fd1a5768ca4dac698491c9898e09e32f`
- Full checkpoint [link](../../../workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth)，SHA `f7df2a16…`
- GCSAM checkpoint [link](../../../workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth)，SHA `acf2fce1…`
