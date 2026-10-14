# UCSD-PTGBM 患者隔离外部确认结果

## 结论

我们没有改变模型，而是把两份早已冻结的内部领先 checkpoint 带到一批患者完全不重叠的 UCSD-PTGBM 病例上各评一次。外部队列为 75 scans / 53 subjects，来自 TCIA 官方主 release；所有在本项目 1,350 例 BraTS 清单中出现过任一时间点的患者均已排除，另列的 BraTS challenge test package 完全未使用。

外部结果的 Scan/Subject/Point 为：

| 冻结模型 | Scan Dice | 95% CI | Subject Dice | 95% CI | Point hit |
|---|---:|---:|---:|---:|---:|
| **GCSAM fit864 epoch184** | **`0.755647338`** | `[0.717080,0.789172]` | **`0.755151670`** | `[0.707873,0.796526]` | **71/75 (`0.946667`)** |
| Full972 random-init voxel-KD epoch168 | `0.745554128` | `[0.704767,0.781595]` | `0.739054558` | `[0.686028,0.784535]` | 70/75 (`0.933333`) |

GCSAM − Full972 KD 为 Scan `+0.010093211`、Subject `+0.016097112`、Point `+1/75`。相应配对 95% CI 为 Scan `[-0.002465,+0.027164]`、Subject `[-0.000631,+0.038954]`，仍跨 0；因此 GCSAM 是该外部 cohort 的三指标点估计领先者，但不能宣称统计显著。

## Idea 与动机

当前复用内部 test270 的 GCSAM 与 Full972 KD Scan 仅差 `0.000244`且置信区间跨 0；fixed validation 与 test 又都已反复使用。近期 LAWA6、Gamma 增强、Lookahead 的跨 seed 失败进一步说明，再在旧集合追一个小数点不足以证明泛化。

一句话直觉：两位在旧题库分数接近的学生，应该去另一批从未见过的患者上各答一次，且不能看完答案再改规则。

## 数据审计与唯一变化

- TCIA 官方 `participants.tsv`：178 subjects / 243 timepoints。
- 官方主 GitHub releases：136 subjects / 184 个唯一 `run-post_T1w` + `BraTS_tumor_seg` 配对。
- 与本项目冻结 1,080 train + 270 reused-test union 做映射后，主 release 有109 scans / 83 subjects 已暴露。
- 患者级严格差集：75 scans / 53 subjects；case overlap=0，subject overlap=0。
- 官方另列的42-subject/59-study BraTS-GLI challenge test package未进入 inventory、下载或评估。
- 唯一变化是评测 cohort。两个 checkpoint、T1 对比增强单通道、Whole Tumor（WT）=`{1,2,3}`且切除腔`4`排除、canonical RAS、N4 shrink2、image-only brain crop、128³、logit阈值0、1×forward、无测试时增强、无 Segment Anything Model、无后处理全部不变。

## 数据工程结果

- 选择性下载 103,182,525 bytes，75对文件全部与官方 asset SHA-256 一致。
- 官方participants表有一处不在主release的local/BraTS subject非双射（`sub-0038_03→BraTS-GLI-02289-102`）；结果后按两个namespace重审，75例与84个已暴露UCSD local subjects的overlap仍为0，异常`sub-0038`未入选，因此现有队列不受影响。
- 75/75 图像与标签均为256³、1.0mm各向同性；canonical后shape/affine匹配且全部finite。
- 原始标签全集精确为`{0,1,2,3,4}`；75例WT均非空，128³最近邻变换没有改变空/非空状态。
- Cache tensor SHA-256：images `ebfb40f3e6c45231926c653174dcbd2494ded74f37154b1d381f6413ade80940`；labels `4d5fcb9de1b484ea570cfc0d22ad37a120e7c4c01049145a8c42f63e3c04dd38`。
- 合成合同9项通过；两 checkpoint 的真实 GPU 零输入 smoke 均输出有限 `[1,1,128,128,128]` logit，参数均4,700,897。
- 结果后完整cache reproduction从75对官方raw重新执行N4与全部预处理；images/labels均`75/75`逐case byte-exact、最大差0，临时751MB目录自动清理，正式cache未修改、模型未运行。机器报告`CACHE_REPRODUCTION_AUDIT.json` SHA-256 `f4f25187af4a1771020657b591b978026e5307f63810eed96eec2f8cd8a9c40c`。

结果后先用已暴露且被external75排除的1个train972病例和1个fixed-validation108病例做格式QC：官方UCSD版为256³，本项目BraTS副本为182×218×182；以共同固定体素偏移`[-37,-19,-37]`对齐后，两例原始分割逐体素完全一致，T1标准化相关为`0.9862/0.9999`，证明ID、模态和标签接线正确。但其中一例官方256³版本保留了更大的非零脑/FOV，导致相同image-only crop算法得到不同裁剪范围和128³标签rasterization。

随后把标签身份审计扩展到全部已暴露development overlap，且在下载前硬排除28个reused test270扫描：81例中76例中心标签bit-exact；5例仅差2–36个体素，WT Dice最低仍`0.999742`，符合官方新release相对BraTS2024做极小边缘修订，而不是ID错配。严格exact gate因此以`76/81`按预设失败并保留原始报告；follow-up只量化差异，没有运行模型。这是外部网格/FOV/版本差异，不能称输入tensor与内部cache完全相同，也不能在见到结果后改裁剪重测。详细审计见`POSTRUN_AUDIT.json`、`DEVELOPMENT_OVERLAP_LABEL_AUDIT.json`和`DEVELOPMENT_OVERLAP_LABEL_DELTA_AUDIT.json`。

## 冻结门与解释

Full972 KD 的绝对 Scan/Subject/Point 分别超过预注册 `0.70/0.70/0.90` 三门，说明它确实能带到未见患者；但相对 GCSAM 的两个点估计门和两个 `CI lower>=-0.003` 门失败。八个性能原子门通过4个、失败4个，aggregate gate失败。

原直觉得到部分支持：独立外部确认是可行且必要的，两模型绝对表现都保持在约`0.75`；但“不满训练集的 GCSAM 只是内部 Scan 小波动、Full972 KD 更稳”的隐含预期没有得到支持。外部点估计反而一致偏向 GCSAM，且幅度远大于内部 `0.000244`，这说明锐度感知更新可能学到了更可迁移的参数区域。不过配对 CI 仍跨0、改善/变差病例接近对半（以 KD−GCSAM 计38/36/1；患者27/25/1），所以这里只支持“值得在未来独立 cohort 再确认”，不支持用本结果调 `rho`、epoch或选择新模型。

结果后描述性敏感性审计进一步限制了解读：GCSAM仅在36/75 scans、25/53 subjects上更高，KD在38/75、27/53上更高；GCSAM获胜时平均`+0.04237`，落后时平均`-0.02022`。单例从KD `0.122704`到GCSAM `0.611937`（`+0.489233`）是主要拉动；仅作post-hoc描述地去掉该例/该单扫描患者，均值差缩至Scan `+0.003618`、Subject `+0.006998`，中位差为`-0.000329`。正式75例结果和预注册CI不变；这表明领先来自少数较大的救援，而不是多数病例普遍改善。证据`DESCRIPTIVE_SENSITIVITY_AUDIT.json`。

用于未来队列设计的粗略正态近似（不是本结果的显著性检验）显示：53个患者配对差SD=`0.07628`，若RHUH只有40人，95%半宽约`0.02364`，大于当前`0.01610`效应；α=0.05、80% power检测同等效应约需177名独立患者。故RHUH40只能承担方向/稳健性复现，未来若跨cohort合成必须事先注册subject-level meta-analysis，不能事后把扫描混池。证据`POWER_PLANNING_AUDIT.json`。

## 成本

- 训练：0 GPU-hour；未改变权重。
- 下载：103.18MB、36.75秒；raw目录约100MB。
- N4 + cache：419.23秒、约751MB。
- 推理：Full972 KD 2.43秒、GCSAM 1.55秒，共150次locator forward；单模型部署仍为1×forward。
- 本次共同进程记录到的推理增量峰值约5.22GB；secondary字段为0是因为同一进程复用了已达到的CUDA high-water mark，不代表零显存。训练显存不新增。
- 当前外部 workspace合计低于1GB；无大cache删除。

## Evidence、访问与边界

- External confirmation已访问一次；这是预注册、患者不重叠、非test-tuned的公开外部cohort确认，但不是服务器隐藏的blind challenge。
- 本实验没有访问新的train、fixed validation、outer、internal confirmation或reused test270；历史模型自身的既有访问边界保持原样。
- UCSD与BraTS采用共同标注体系且官方映射揭示部分数据重叠；严格差集解决了本项目可见患者泄漏，但仍应标注为同生态、术后单中心外部发布队列，而不是完全不同来源的临床多中心证据。
- Grid/FOV caveat：官方UCSD为256³ MNI发布网格，本项目训练用BraTS副本为182×218×182；同算法的image-only crop/128³ rasterization并不保证tensor等价。该域差异包含在本次一次性外部结果中。
- External结果不得回流选择checkpoint、阈值、预处理、病例、ensemble或后处理，也不得在本队列上重测修改版。

## 证据哈希

- Raw external report：`workspace/BraTS/ucsd_ptgbm_external_confirmation_20260902/UCSD_EXTERNAL_CONFIRMATION.json`，SHA-256 `91910760fbab804ee17b3ad8027a035a6676b02307a6f34d8408ba858c42f593`。
- Preaccess receipt：`PREACCESS_RECEIPT.json`，SHA-256 `c13513df4780dceb9c64ea1509985d54cb32ff3b4364b979f895fabc14c1da8b`。
- Official asset inventory：`OFFICIAL_ASSET_INVENTORY.json`，SHA-256 `6cb1f5a5eb740c06c6e8b01e57c7e6aab6dd8a7472f02d8acdd42540e2d16c38`。
- Download receipt：`workspace/BraTS/ucsd_ptgbm_external_confirmation_20260902/raw/DOWNLOAD_RECEIPT.json`，SHA-256 `4d635a4b92da11250381b479d1ebc954062197045865a059f88915af5111b331`。
- Cache index：`workspace/BraTS/ucsd_ptgbm_external_confirmation_20260902/cache128/index.json`，SHA-256 `14d944f945d5f6396ab5f676dffeb172be94dd7d781098b615835775640698b8`。
- 完整证据链验证器：`scripts/verify_ucsd_ptgbm_external_confirmation.py`，验证75例/53人零重叠、缓存与报告SHA、per-case重算、External榜逐字段一致、development-only标签身份审计，最终passed；收据`VERIFICATION_RECEIPT.json`。
- Cache全量重建：`scripts/verify_ucsd_ptgbm_cache_reproduction.py`，runner SHA-256 `8b120c762745d487b25431684d675f46573f11fdc3185221f770ee0e5dc5bf62`；1项tensor比较测试通过。
- GCSAM checkpoint SHA-256 `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`。
- Full972 KD checkpoint SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`。

## 下一决定

关闭本 cohort，不做任何结果依赖的重跑。保留 GCSAM 为 UCSD external75 的单模型点估计第1，Full972 KD 保留为内部reused test270的Subject/Point第1；两种cohort绝不混排。若未来获得第二个真正患者独立的多中心人工标签队列，只允许原样确认两份冻结模型，或对任何新模型使用全新、未见的病例。

## 一手来源

- [TCIA UCSD-PTGBM collection](https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/)
- [Scientific Data数据论文（DOI 10.1038/s41597-025-06499-z）](https://www.nature.com/articles/s41597-025-06499-z)
- [TCIA官方GitHub逐文件release](https://github.com/data-nih/tcia/releases)
