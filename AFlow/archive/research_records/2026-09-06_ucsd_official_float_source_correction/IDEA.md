# Experiment idea: UCSD75官方float32来源纠正

Status: completed once and closed; preregistration below was frozen before official-float cache construction and model inference

## 1. 现在的问题是什么？

旧UCSD75外部评测覆盖75 scans / 53 subjects，GCSAM184的Scan/Subject Dice/Point为`0.755647/0.755152/71`，Full972 voxel knowledge distillation（体素知识蒸馏，KD）epoch168为`0.745554/0.739055/70`。来源复核却发现旧输入来自Pittsburgh Fiber Data Hub整理的uint8镜像，不是TCIA官方float32影像：官方与镜像图像`0/75`逐体素相同，18,156个官方非零体素在镜像里变为0；标签仍`75/75`逐体素相同、图像和标签几何`150/150`一致。因此旧分数只能代表镜像8位版，不能代表官方原始浮点版。

## 2. 一句话直觉

把同一批试卷从压缩扫描件换回原件，再让同两位已经冻结的阅卷员按完全相同规则重判；差异只说明扫描件压缩的影响，不是又找到了一批新考生。

## 3. 为什么选择这个idea？

- UTSW362官方人工标签出现发布材料未定义的原始值3和5，不能靠猜测把它们并入或排除Whole Tumor（全肿瘤，WT），所以该新队列在模型前关闭。
- UCSD75的官方患者映射、75对图像/标签和逐文件哈希已经核验；官方标签与旧标签完全一致，唯一未回答的问题是float32影像经过同一完整输入链后是否改变模型结果。
- TCIA官方UCSD-PTGBM集合是一手来源；已有来源审计保存官方包1019清单、临床v3映射和直接下载收据。
- 旧镜像结果没有否定本纠正，因为N4 bias-field correction（N4偏置场校正）的拟合、非零支持裁剪和标准化都可能对量化敏感；反之，高灰度相关也不能证明最终128³输入或模型输出相同。

## 4. 具体怎么做？

输入是已冻结的同75例官方T1 post-contrast（T1增强）MRI和同一官方分割标签。每例先转canonical RAS方向，运行与旧协议相同的N4（shrink factor 2），用校正后正值影像的非零包围盒裁剪，在源非零脑区做population z-score，再以三线性影像/nearest-exact标签缩放到`128×128×128`。标签按旧合同把原值`{1,2,3}`作为WT，值4切除腔排除。

缓存通过全部工程门后，先运行已冻结Full972 KD168，再运行已冻结GCSAM184；两者都以logit `>0`输出mask，以最高logit位置输出自动point。每模型identity视图一次，无test-time augmentation（测试时增强，TTA）、无Segment Anything Model（SAM）、无后处理、无集成。计算Scan Dice、Subject Dice、正WT扫描Point hit及固定seed的病例/患者bootstrap 95%区间；同时按病例与旧uint8报告做描述性配对差值。

这是来源纠正，不训练teacher或student，不把结果用于模型、阈值、病例或预处理选择。

## 5. 唯一变化与固定项

- 唯一变化：同75例T1增强影像的文件字节由旧uint8整理镜像换成TCIA包1019直接取得的官方float32 NIfTI。
- 固定不变：75 scans / 53 subjects及顺序、患者排除规则、官方标签数组和几何、WT=`{1,2,3}`且排除4、canonical RAS、N4 shrink2、image-only正值包围盒、源非零population z-score、128³、插值、两个checkpoint及SHA、阈值0、identity单次forward、无TTA/SAM/后处理/ensemble、bootstrap样本数和seed、指标定义。
- 旧镜像cache、raw report与归档不可修改；官方版建立新cache和新报告。

## 6. 成功、失败和停止规则

- Primary：分别报告两模型官方float32版Scan/Subject Dice与Point hit；另报告官方版减旧镜像版的同模型配对差值和95% bootstrap区间。
- 工程放行门：75/75病例与顺序完全匹配；150/150源文件SHA匹配已冻结直接下载收据；75/75图像/标签canonical几何一致；label值域仅`{0,1,2,3,4}`；官方标签和旧镜像源标签75/75 exact；缓存全finite、75例empty状态不因resize改变；独立重建至少1例tensor exact；synthetic GPU smoke通过；缓存和checkpoint哈希与预访问收据相符。
- 解释门：结果无论升、降或持平都归档；不设置性能晋级门，不据此调任何因素，不做结果依赖重跑。只允许在模型访问前修复实现错误，并写amendment；模型一旦开始，不重复。
- Failure/stop：任一工程门失败即0模型forward停止。若模型运行时非人为中断且产出不完整，记录失败边界，不用同一cohort试第二套实现。
- 是否允许进入test：不进入任何新test。本次就是对已经访问过的同一UCSD75重复来源纠正，必须标为reused、not pristine blind、not independent new cohort；不访问内部test270、outer或confirmation。

## 7. 成本与风险

- 训练时间：0。
- 峰值显存：预期约旧评测水平，远低于单张A100 80GiB；正式记录实测值。
- 磁盘：官方源约407MiB已存在；新cache约751MiB，放`/tmp`以避免仅9.5GiB可用的workspace继续膨胀；报告很小。
- 推理forwards/延迟：75例×2模型×identity=`150` forwards；部署仍各模型1×。
- 最大技术风险：uint8量化不仅改变灰度，还让18,156个非零体素变0，可能先改变N4拟合与crop边界，造成连锁差异；因此必须比较中间cache而不能只比较源相关系数。

## 8. 最终结果

- Official-float：GCSAM184 Scan/Subject/Point=`0.755890/0.755312/70`；Full972 KD168=`0.746255/0.740165/70`。
- 旧mirror-uint8 reference：GCSAM184 `0.755647/0.755152/71`；Full972 KD168 `0.745554/0.739055/70`。
- Official−mirror：GCSAM=`+0.000242/+0.000160/−1`，Scan/Subject 95% CI=`[-0.001168,0.001658]/[-0.000869,0.001170]`；Full972=`+0.000701/+0.001110/0`，CI=`[-0.000863,0.002658]/[-0.000735,0.003806]`。四个Dice区间都跨0。
- 是否支持原直觉：部分支持。影像cache 0/75 exact，说明量化差异穿过预处理；但两模型聚合Dice只小幅变化且不确定，GCSAM Point仍改变1例。
- 下一步：已关闭UCSD75重复评测，不回流选择；旧mirror和新official行按来源共同保留。
- 报告/归档/哈希：[RESULT.md](RESULT.md)、[RESULT.json](RESULT.json)、raw SHA-256 `a1155914b16b648f97f11a7d7a20cb27096d7eed9f9662db73f4905860da04d2`。

## 一手来源与冻结证据

- TCIA UCSD-PTGBM collection: <https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/>
- 官方来源更正：[`UCSD_SOURCE_RESULT.md`](../../reviews/2026-09-05_30h_chain_audit/UCSD_SOURCE_RESULT.md)
- 全75例源数组比较SHA-256：`1dc05453fce200e11739dafbc1b73d33a5ef1bd491d3174b3ebb42504d101ac7`
- 旧raw report SHA-256：`91910760fbab804ee17b3ad8027a035a6676b02307a6f34d8408ba858c42f593`
- Full972 KD168 checkpoint SHA-256：`f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`
- GCSAM184 checkpoint SHA-256：`acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`
