# Experiment idea: BraTS-Africa95冻结baseline外部评测

Status: completed once and closed; preregistration below was frozen before package payload, NIfTI arrays, label values, cache or model access

## 1. 现在的问题是什么？

当前内部reused test270总体点估计最高是Full972 voxel knowledge distillation（体素知识蒸馏，KD）epoch168的固定Flip8 test-time augmentation（八镜像测试时增强，TTA），Scan/Subject Dice/Point=`0.757270/0.741254/265`，但该test已多次揭示。UCSD官方float32结果又是同一75例的来源纠正，不是第二个独立队列。UTSW362、EGD374、UCSF-PDGM分别卡在未定义标签3/5、机构签署Data Usage Agreement、与BraTS患者重叠无法排除。项目缺少来自不同医疗系统、专家标注、公开可取的新外推证据。

## 2. 一句话直觉

把两位已封卷的阅卷员和当前“把试卷从8个镜像方向看完再平均”的正式方案，带到尼日利亚六家医院考同一门新试卷；考完不能改答案规则。

## 3. 为什么选择这个idea？

- TCIA BraTS-Africa package948只有约1.6GB；完整目录审计在payload前得到146例×5文件，其中发布者`95_Glioma`目录恰有95例，每例唯一T1CE和专家seg。51例`OtherNeoplasms`由发布者类别事先排除，不看模型。
- 一手论文/TCIA说明这些1.5T、多中心、较厚原始层面的MRI经过BraTS标准配准、N4偏置场校正、1mm重采样和脑提取；nnU-Net只做预标，最终由有5年以上肿瘤分割经验的美国认证神经放射科医师人工修订，适合检验低资源域迁移。
- 六个尼日利亚中心不在BraTS2024 post-treatment的Duke/UCSF/Missouri/UCSD/Heidelberg/Michigan/Indiana七站。官方ID没有跨库患者桥，所以只称institution-disjoint、无已知重叠，不声称用ID绝对证明patient-disjoint。
- UTSW/EGD/UCSF的负结果是来源语义、权限或身份门失败，没有运行能否迁移到BraTS-Africa的模型，不能否定本队列。
- 一手来源：[TCIA collection](https://www.cancerimagingarchive.net/collection/brats-africa/)、[challenge paper](https://doi.org/10.1093/noajnl/vdag082)、[dataset paper](https://doi.org/10.1148/ryai.240528)。

## 4. 具体怎么做？

先对冻结的95例官方T1CE与seg共190路径做两份独立TCIA传输；由于package目录无逐文件checksum，只有两份路径/bytes/SHA-256逐项相同才放行。固定首例先解码，95例全量随后检查图像/标签canonical几何一致、finite、标签为整数且只在`{0,1,2,3,4}`、每例WT非空。

输入T1CE经canonical RAS、与项目训练相同的N4 shrink factor 2、校正后正值影像包围盒裁剪、源非零脑区population z-score、三线性影像/nearest-exact标签缩放到128³。BraTS-Africa是术前数据，没有resection cavity（切除腔）；WT冻结为seg所有正值的union，不需要在结果后猜三个肿瘤子区的具体编号。

模型一是Full972 KD168：一次循环固定运行三轴反射群全部8个视图，identity输出单列，8个logit逆翻后float32等权平均作为Flip8。模型二是GCSAM184，只运行identity。所有输出以logit `>0`得到WT mask，以最高logit位置作为自动point；无Segment Anything Model（SAM）、无其他后处理/ensemble。计算95 scans / 95 subjects的Scan/Subject Dice、Point hit及固定10,000次percentile bootstrap；比较Full Flip8−identity，以及两个identity模型。没有训练、teacher或loss。

## 5. 唯一变化与固定项

- 唯一变化：新增发布者`95_Glioma`目录的BraTS-Africa公开post-challenge外部cohort。
- 固定：95 cases/95 subjects及顺序；官方T1CE+seg；preoperative WT=all positive；canonical RAS、N4 shrink2、image-only crop、source nonzero z-score、128³和插值；Full972 KD168与GCSAM184 checkpoint及SHA；阈值0；Full的identity/固定Flip8公式；GCSAM identity；无SAM/后处理/模型ensemble；指标、empty policy、bootstrap次数/seed。
- 不包含`51_OtherNeoplasms`；不按病理、中心、增强、图像质量、标签体积或模型结果删除病例。
- 原挑战角色为60 train/15 validation/20 test，但TCIA现已公开全部95标签。本项目不使用其中任何部分训练/调参，合并报告为public post-challenge external95；不是隐藏blind test。

## 6. 成功、失败和停止规则

- Primary结果：Full Flip8与Full identity的Scan/Subject Dice及Point差；同时报告GCSAM identity与Full identity的单模型比较。绝对三行都需报告，不因输赢省略。
- 工程门：package/cohort/inventory哈希exact；190路径双传输文件存在、正大小、路径/bytes/SHA逐项exact；95/95 T1CE/seg canonical geometry在`atol=1e-5`内相同；图像finite且正值支持非空；标签finite/integer、值域⊆`{0,1,2,3,4}`、background0与positive WT同时存在；cache全finite；resize不改变WT empty状态；固定首例独立cache重建image/label byte exact；synthetic GPU smoke、checkpoint/cache/preaccess哈希全过。
- 没有性能promotion gate：任何正、负、tie都一次性归档并关闭；不能据结果调checkpoint、阈值、N4、crop、病例、view、TTA权重或后处理，不能重跑同cohort。
- 工程失败：在相应访问边界停止。模型开始后若非人为故障未完整产出，保存失败边界，不换实现获取第二套结果。
- 不进入内部test270/fixed validation/outer/confirmation/UCSD/RHUH/UTSW。

## 7. 成本与风险

- 训练时间：0。
- GPU：预计单A100峰值<8GiB；正式共Full 95×8 + GCSAM 95×1=`855` forwards。
- 磁盘：双源每份277,630,584 bytes，共约530MiB；95例cache约950MiB，全部放`/tmp`；归档仅保留代码/收据/报告和cache index。
- CPU：N4预计15–25分钟；下载预计数分钟；GPU推理预计一分钟内，实测另报。
- 部署：Full Flip8仍8×，两个identity仍1×；不增加参数或组件。
- 最大风险：原始1.5T扫描常为3–5mm层厚，官方重采样到1mm不会恢复丢失的真实细节；结果必须标grid/rasterization caveat。另只能证明机构/国家来源差异，不能用共同ID桥绝对排除患者。

## 8. 最终结果

- GCSAM184 identity=`0.591619/0.591619/84/95`；Full972 KD168 identity=`0.523003/0.523003/87/95`；Full Flip8=`0.526438/0.526438/88/95`。
- GCSAM−Full identity Dice=`+0.068617`，95% CI=`[+0.044472,+0.092498]`，Point`−3`；80/95改善。Flip8−identity Dice=`+0.003435`，CI=`[-0.010536,+0.016815]`，Point`+1`。
- 是否支持直觉：支持该新域能暴露真实泛化差异；GCSAM区域Dice更稳但Point更低，Flip8的内部收益未稳定跨域。三模型都显著低估WT范围。
- 来源更正：实际95_Glioma由package datainfo映射为四中心，不是raw冻结文字的六中心；分数不变。
- 下一步：同cohort关闭，不回流选择或重跑。
- 报告：[RESULT.md](RESULT.md)、[RESULT.json](RESULT.json)、raw SHA `c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3`。
