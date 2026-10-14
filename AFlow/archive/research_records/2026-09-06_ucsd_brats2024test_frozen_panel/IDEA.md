# Experiment idea: UCSD BraTS-GLI 2024公开test包冻结checkpoint面板（56 scans / 41 subjects）

Status: frozen before any pixel access

## 1. 现在的问题是什么？

项目当前六个已揭示cohort（internal270、UCSD75、Africa95、CPTAC33、WAW-pre152、WAW-post31）里，治疗后（post-treatment）阶段的外部证据只有WAW-post31：31人、Flip8−identity=`+0.007312`但CI `[-0.001925,+0.016626]`跨0，GCSAM−Full=`+0.001657`同样跨0，全部positive-but-uncertain。其余外部全是术前，与训练（BraTS2024 post-treatment）阶段不匹配。同时内部fresh容量只剩2人/3 scans，无法建立新内部证据。我们需要一份**新患者、治疗后阶段、专家批准标签**的公开队列来缩窄Flip8与GCSAM方向的不确定性。

## 2. 一句话直觉

同一批冻结"医生"（两个checkpoint）再去考一场**从未见过的患者、而且考题类型（治疗后GBM、同一标签词典）与他们平时训练完全一致**的公开考试；考完只记分数，不许回头改答案。

## 3. 为什么选择这个idea？

- 现有结果暴露的现象：Flip8五cohort Scan点估计全正但只有internal/CPTAC/WAW-pre三处CI全正；GCSAM只有Africa CI全正。treatment-stage-matched外部样本不足是主要缺口（见NEXT_EXTERNAL_DATA_SPEC.md E3）。
- 一手来源：TCIA UCSD-PTGBM集合**Version 2（2026-01-22）公开了原BraTS-GLI 2024隐藏test中UCSD贡献的部分**（package 1221，"UCSD-PTGBM-BraTS-2024-test-set"，59 studies/42 subjects，每study有T1post+BraTS_tumor_seg，CC BY 4.0）；Version 3（2026-03-13）修正了临床表患者ID。2026-09-02本项目UCSD75评测时该包被明确排除、从未访问。
- 患者独立性可证明：官方v3临床表提供逐study的BraTS case ID映射。any-timepoint规则下42人中仅`UCSD-PTGBM-0038`（映射`BraTS-GLI-02289`）出现在项目1,350例训练union，也在先前main release 136人中；排除后41人/56 scans与union（613 subjects）、已评UCSD75（53 subjects）零重叠。全表唯一跨ID歧义（02289↔0036/0038）也随0038排除而消除。
- 为什么现有负结果没有否定它：这是新患者评测，不是重测。此前"同一UCSD75第三次评测禁止"针对同75例；本cohort与其零患者重叠。OpenNeuro/MOTUM的失败是输入支持门问题，UCSD official float32来源在2026-09-06已被证明可走通完整canonical链（同集合、同发布者、同格式）。

## 4. 具体怎么做？

输入：每study官方`T1post.nii.gz`（T1对比增强）与`BraTS_tumor_seg.nii.gz`。包内无`.sums`，故两次独立官方Aspera传输逐文件SHA-256必须一致（Africa95/MOTUM先例），gzip容器CRC由完整解码隐式检验。逐例：canonical RAS→N4（shrink2、4×50迭代、线程1）→image-only正值包围盒crop→源非零z-score→三线性128³影像/最近邻128³标签→脑mask清零。WT=raw labels `{1,2,3}`的并集，排除切除腔`4`。模型：Full972 KD168输出identity（1×）与固定Flip8（8种三轴reflection逆翻后float32等权raw-logit平均，8×）；GCSAM184输出identity。阈值0出mask，全体素最高raw-logit为自动point。无SAM、无后处理、无模型ensemble。

## 5. 唯一变化与固定项

- 唯一变化：新增UCSD-BraTS2024test公开cohort（56 scans/41 subjects，冻结于SOURCE_AUDIT_V2）。
- 固定不变：两个checkpoint（SHA `f7df2a16…`/`acf2fce1…`）及其epoch、canonical N4预处理链全部参数、WT定义、阈值0、视图公式与顺序、无后处理、指标定义（Scan Dice等权、Subject Dice先患者内平均、Point hit阳性分母）、bootstrap（percentile、10,000次、seed 20261601起）、解释门。不因结果调整任何设置、不重跑。

## 6. 成功、失败和停止规则

- Primary：`Full Flip8 − Full identity`与`GCSAM identity − Full identity`的Scan Dice配对差及95% CI（scan级配对bootstrap；subject级同报）。
- 解释门（性能前冻结）：delta>0且scan CI下界>0 → `supported_on_ucsd_brats2024test56`；delta>0但CI跨0 → `positive_but_uncertain`；delta≤0 → `not_supported_or_reversed`。无performance promotion门；结果无论正负一次归档。
- 工程停止门（任一失败即模型前停止，不放宽、不换例）：双传输112文件SHA全exact；每例image/label native shape exact、affine误差≤1e-5、raw label值⊆{0,1,2,3,4}；raw WT（labels 1/2/3）为空的scan在性能前按确定性规则排除并如实报告（本队列若出现，仅影响该scan，不重选cohort）；零WT crop丢失门（raw WT必须100%落在image-positive bbox内）；首例probe与全量worker同线程设置且byte exact；cache全finite/binary；synthetic smoke通过；MODEL_PREACCESS在cache后、模型前冻结。
- 是否允许进入test：本实验本身就是一次性外部评测；internal test270/fixed validation/outer/confirmation/其他已关闭cohort一律不访问。评测后禁止用结果选择checkpoint、阈值、视图、N4、病例或后处理。

## 7. 成本与风险

- 训练：0。
- 下载：112文件×2次独立传输≈624MB（copy A存workspace，copy B存/tmp）。
- cache：56×128³×(4+1)bytes≈587MB，建在/tmp（661G可用）；含N4约10–15分钟CPU。
- 推理：56×9=504 forwards（若无空WT排除），预计<10秒、峰值约5.1GiB单卡。
- 磁盘gate：下载/建cache前重查`df`，repo保留≥1GiB余量。
- 最大技术风险：(1) BraTS2024 test标签可能有空WT scan（规则已冻结）；(2) 无官方checksum，双传输不一致则停止；(3) 该cohort与训练同为UCSD+BraTS流程，域差异小，可能出现接近internal的分数——解释时必须标明"patient-disjoint but not institution-disjoint、not hidden blind"，不得当作跨机构泛化证据。

## 8. 最终结果（完成后填写）

- 见RESULT.md。
