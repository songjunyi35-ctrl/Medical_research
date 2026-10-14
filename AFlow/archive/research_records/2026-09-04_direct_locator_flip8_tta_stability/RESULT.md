# Direct locator固定Flip8测试时增强：最终结果

状态：完成；两个fresh inner复制与唯一一次reused internal test270均已执行。Flip8成为test270新的Scan/Subject/Point全面点估计第1，但证据是post-test development、非pristine blind、非外部确认。

## 初学者版结论

**一句话idea：** 同一位“医生”把一张MRI按8种镜像方向都看一遍，再把判断翻回原方向平均，以抵消摆放方向造成的偶然误差。

**为什么做：** Full972知识蒸馏（Knowledge Distillation, KD）epoch168在旧test270的Scan Dice为`0.751760`，只比GCSAM榜首`0.752003`低`0.000244`，而Subject/Point更高；训练又已经随机见过全部三轴翻转，所以固定完整8视图是一个不改训练、只检查推理等变误差的单因素方法。

**只改变什么：** identity单次前向改为全部8种三轴reflection依次前向，输出逆翻回原坐标后按固定顺序做float32 raw-logit均值。模型、checkpoint、T1ce、WT=`{1,2,3}`、true128³ image-only brain crop、阈值0、Point=argmax、无Segment Anything Model、无连通域/形态学/后处理全部不变。

## Stage A：两个fresh inner复制

- Replicate A，108 scans/47 subjects：Flip8−Identity Scan/Subject/Point=`+0.006667794/+0.006930161/-1`；10,000次配对bootstrap 95% CI=`[+0.003718,+0.009444]`/`[+0.003622,+0.010338]`。
- Replicate B，108 scans/49 subjects：`+0.006988400/+0.007332868/+3`；CI=`[+0.004026,+0.010414]`/`[+0.003229,+0.011789]`。
- 两复制平均Scan/Subject提升=`+0.006828097/+0.007131514`，远高于冻结的榜首差`0.000243638`；全部门通过。
- 两个模型都先完成固定168轮，冻结`INNER_PREACCESS`后才访问inner；训练期间没有inner指标。

## 唯一一次reused test270

同一Full972 KD168 checkpoint的identity被重新计算，并与旧报告的完整summary和270条per-case记录完全相等，证明候选差异不是接线、batch或浮点评测伪差异。

| 方法 | Scan Dice | Subject Dice | Point hit | forwards |
|---|---:|---:|---:|---:|
| identity | `0.751759537` | `0.737327107` | 262/270 (`0.970370`) | 1× |
| **固定Flip8** | **`0.757269662`** | **`0.741253585`** | **265/270 (`0.981481`)** | 8× |

Flip8−identity=`+0.005510124/+0.003926478/+3 points`。Scan配对95% CI为`[+0.002751,+0.008022]`，整体高于0；Subject配对CI为`[-0.000379,+0.007354]`，跨0，因此只说Subject点估计提高，不称显著。217/270 scans改善、53下降；96/119 subjects改善、23下降。

相对此前Scan榜首GCSAM fit864 epoch184，Flip8点估计高`+0.005266486` Scan、`+0.003987165` Subject、`+13` Point hits。没有冻结二者间的配对检验，因此不声称相对GCSAM显著。

冻结seed20260817的描述性绝对95% CI：Flip8 Scan `[0.739089,0.774152]`、Subject `[0.714109,0.766152]`；identity Scan `[0.734140,0.768644]`、Subject `[0.711486,0.761430]`。范围是270 scans/119 subjects。

## 实现偏差与失败边界

raw report的绝对CI helper实际标注seed20260826，而pretest receipt冻结seed20260817；这是实现偏差，不能隐藏。它不影响mask、Point、任何点估计、使用正确seed20261207的配对CI或成功门。raw report保持原样；只从其已保存per-case行离线重算一次seed20260817绝对CI，没有模型推理、test cache访问、调参或重测。修正证据见[BOOTSTRAP_CORRECTION.json](BOOTSTRAP_CORRECTION.json)。

## 代价、解释与下一步

- Stage A双A100墙钟约3.404h、共6.783 GPU-hours；1,728次locator forward。test270增加2,160次forward。
- 部署仍是同一个4.70M参数、同一个56.5MB checkpoint，峰值接近单视图，但计算/延迟约8×。
- 本实验workspace约55MB，归档在写结果文档前约348KB；没有创建新cache，也没有删除材料。
- 直觉得到支持：两个fresh inner和reused test三处都同向提高Scan/Subject，test Point也增加3；方向平均确实消除了可复现的等变误差。
- test270已经访问且不得再用于视图、权重、阈值、checkpoint、后处理或重测。fixed validation108、outer、confirmation和UCSD75在本实验中未访问。
- 下一步需要新的独立外部cohort验证固定Flip8；在此之前只能称“reused internal test270全面点估计第1”，不能称隐藏榜SOTA或普遍优越。

## 证据

- [七问与方法解释](IDEA.md)
- [冻结预注册](PREREG.md)
- [Stage A raw](STAGE_A_INNER.json)（SHA-256 `d516f3ad…6dcff68`）
- [inner preaccess](INNER_PREACCESS.json)（SHA-256 `b2e3cf16…4370a4f`）
- [test实现凭证](TEST270_IMPLEMENTATION_RECEIPT.json)（SHA-256 `cae62293…07d244`）
- [test preaccess](TEST270_PREACCESS.json)（SHA-256 `9ee84944…9b1609`）
- [test started](TEST270_STARTED.json)（SHA-256 `04e4b055…dba40`）
- [raw test report](RAW_TEST270.json)（SHA-256 `29efc557…f45ad`）
- [机器结果](RESULT.json)
- [部署checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth)（SHA-256 `f7df2a16…80d17b`）
