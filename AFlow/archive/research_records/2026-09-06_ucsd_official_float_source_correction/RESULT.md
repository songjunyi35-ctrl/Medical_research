# UCSD75官方float32来源纠正结果

状态：✅ 2026-09-06按预注册完成一次性来源纠正；同一reused cohort已关闭，不调参、不重跑。

## 初学者先看结论

旧UCSD75像是同一批MRI的“8位压缩扫描件”。这次把影像换回TCIA直接发布的float32“原件”，同两份冻结模型、同75 scans / 53 subjects、同标签和同一处理规则各跑一次。结果说明：对这两个模型而言，压缩造成的平均Dice变化很小，两个来源差值的95%置信区间都跨0；但GCSAM184的自动点少命中1例，所以不能说输入字节完全无影响。

官方float32版上，GCSAM184的Scan/Subject Dice点估计仍高于Full972 KD168，但两模型配对Dice区间仍跨0，Point同为70/75。只能称它是**这批重复UCSD75、official-float32来源版本的Scan/Subject点估计第1**，不能称显著更好，也不是一份新的独立外部验证。

## 问题、直觉与唯一变化

旧mirror-uint8结果为：GCSAM184 `0.755647/0.755152/71`，Full972 KD168 `0.745554/0.739055/70`。来源审计发现官方图像75/75为float32、镜像75/75为uint8，源图0/75数组相同且18,156个官方非零体素在镜像中变0；源标签却75/75逐体素相同，150/150图像/标签几何一致。

一句话直觉是：把同一批试卷从压缩扫描件换回原件，再让同两位阅卷员按原规则重判。唯一变化是T1增强影像的来源字节；病例/患者及顺序、官方标签、Whole Tumor（全肿瘤，WT）定义`{1,2,3}`且排除4、canonical RAS、N4偏置场校正shrink2、image-only crop、非零区population z-score、128³、两个checkpoint、阈值0、identity 1×、无测试时增强/Segment Anything Model/后处理/集成、指标和bootstrap全固定。

## 工程结果

- 75映射/53患者/150官方文件SHA全量绑定通过；官方源总计426,129,798 bytes。
- 8-worker完整cache构建耗时868.809秒；首例独立第二次N4/裁剪/缩放后image与label均字节exact，image最大误差0。
- 新旧最终128³ cache：crop origin 75/75相同、crop extent 75/75相同、label 75/75 exact且0个不同体素；image 0/75 exact，每例mean absolute error中位数`0.00419694`，全局max absolute error`7.0779233`。
- 因此模型比较确实只改变影像张量，未改变病例、裁剪框或评测标签网格。
- 首次metadata gate曾因新builder写错冻结状态literal而在NIfTI/cache/GPU/model前停止；Amendment 01只改为实际状态`completed_source_identity_audit`，随后6项新测试和全量源绑定通过。这是模型前实现失败，不影响正式一次性结果。

## 官方float32绝对结果

| 模型 | Scan Dice（95% CI） | Subject Dice（95% CI） | Point hit | forwards |
|---|---:|---:|---:|---:|
| GCSAM fit864 inner-selected epoch184 | **`0.755889505`** `[0.717191,0.789575]` | **`0.755311886`** `[0.707990,0.796823]` | `70/75` (`0.933333`) | 75 |
| Full972 random-init voxel-KD epoch168 | `0.746255043` `[0.706178,0.782003]` | `0.740164863` `[0.688098,0.784670]` | `70/75` (`0.933333`) | 75 |

这里Scan区间按75 scans等权percentile bootstrap，Subject区间先在同患者内平均、再按53 subjects bootstrap；固定10,000次。Point只报告计数/比例，没有事后补置信区间。

official-float32上GCSAM − Full972：Scan=`+0.009634462`，Subject=`+0.015147023`，Point=`0`；配对95% CI分别为`[-0.002163,+0.025053]`和`[-0.000553,+0.035820]`。两个区间跨0，所以只报告点估计方向，不宣称显著优于Full972。

## 同模型的来源影响

| 模型 | official-float32 − mirror-uint8 Scan | 95% CI | Subject | 95% CI | Point差 |
|---|---:|---:|---:|---:|---:|
| GCSAM184 | `+0.000242167` | `[-0.001168,+0.001658]` | `+0.000160216` | `[-0.000869,+0.001170]` | `−1` |
| Full972 KD168 | `+0.000700915` | `[-0.000863,+0.002658]` | `+0.001110305` | `[-0.000735,+0.003806]` | `0` |

GCSAM在40/75 scans改善、34例变差、1例tie；28/53 subjects改善、24人变差、1人tie。Full972为39/35/1 scans与28/24/1 subjects。方向分布接近平衡，且四个来源配对Dice区间都跨0。

## 直觉是否成立、能排除什么

部分支持。量化差异确实穿过了N4和标准化，最终影像75/75不同；但在这两个冻结模型上，平均Scan/Subject Dice变化只有约`0.0002–0.0011`且区间跨0，说明**当前两个模型的聚合Dice对这次具体uint8量化较稳健**。GCSAM Point从71降到70，说明“预测完全不变”不成立。

本结果只排除“旧mirror分数必然因float32原件而发生大的聚合Dice改写”这一担忧；不证明所有模型、阈值、N4变体、分辨率或病例都不受量化影响。也不能把同一75例的第二种来源当作第二次独立复现。

## 成本、访问和下一决定

- 训练：0；teacher/student/权重均不变。
- CPU：cache 868.809秒；源约406.4MiB已存在，新cache约751MiB，保存在`/tmp`；归档保留cache index和整文件哈希，cache可由官方源与脚本重建。
- GPU：两个模型共150 forwards；正式模型计算记录2.733+1.628秒，完整进程墙钟14.584秒。进程观测峰值5,218,738,176 bytes（约4.86GiB）；第二模型的历史增量字段因未重置peak显示0，不单独解释。
- 推理/部署：每个模型仍1×，没有新增部署组件。
- 访问：UCSD75官方float模型访问=true；这是同一已访问cohort的reused source correction、not pristine blind、not independent new cohort。内部test270、fixed validation108、outer、confirmation、BraTS challenge test package均未访问。
- 下一决定：关闭UCSD75重复评测；不根据本结果改checkpoint、阈值、N4、crop、病例、TTA、ensemble或后处理。旧mirror两行与新official两行均保留在external UCSD小节并按来源标注。

## 证据与哈希

- [七问/冻结想法](IDEA.md)、[机器预注册](PREREGISTRATION.json)
- [模型前收据](MODEL_PREACCESS.json)，SHA-256 `299d112a7b032dd7501b07d86dc2607cdd9b05aeef65bce9e388dd9c352a1bec`
- [raw official-float report](RAW_OFFICIAL_FLOAT.json)，SHA-256 `a1155914b16b648f97f11a7d7a20cb27096d7eed9f9662db73f4905860da04d2`
- [started receipt](RAW_OFFICIAL_FLOAT.json.started.json)，SHA-256 `553cdd5bb8395638698d14068ca4fda456a45f2c4fca5976661e2dc8e09125c1`
- [cache index归档](CACHE_INDEX.json)，SHA-256 `2540d5015008e22b40f3f94cc3f33b0d96cc12ee70a14b12c6d15a48e5a0856f`
- [新旧cache比较](CACHE_COMPARISON.json)，SHA-256 `641aba1bf001e22c68b62a1f2d6cfab2b96c17b29cb4e15d969a7b28c2decb73`
- [独立重算收据](VERIFICATION_RECEIPT.json)，SHA-256 `d1eb141f726e998d83c8256a6dae542526d0304b008be723700e6cd0a322decb`
- [历史mirror raw report](../../../workspace/BraTS/ucsd_ptgbm_external_confirmation_20260902/UCSD_EXTERNAL_CONFIRMATION.json)，SHA-256 `91910760fbab804ee17b3ad8027a035a6676b02307a6f34d8408ba858c42f593`
- Full972 KD168 checkpoint：[`plain_fulltrajectory_full972_candidate_epoch168.pth`](../../../workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth)，SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`
- GCSAM184 checkpoint：[`post100_best.pth`](../../../workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth)，SHA-256 `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`
- [canonical leaderboard](../../../AFLOW_BRATS_LEADERBOARD.md)同时保留official-float与mirror-uint8四行。
