# Experiment idea: 保留原始子区标签的fit-only数据工程

Status: planned engineering only; 不启动新模型、不更改正在运行的图像模糊实验。

## 1. 现在的问题是什么？

旧对照在dev180/78的漏画体素约91%—92%来自原标签2，切除腔占错误最高点约38%—65%。当前训练加载器只有合并后的全肿瘤标签和切除腔标签，无法直接研究前景内部1/2/3不同区域的编码。新旋转已正式负结果，不能据此认定达到上限。

## 2. 一句话直觉

保留彩色原稿，不提前只存黑白轮廓；这样以后才能比较不同的区域教学方式。

## 3. 为什么选择这个idea？

一手实现 https://raw.githubusercontent.com/rachitsaluja/BraTS-2024-Metrics/main/metrics_GLI.py 明确本数据原始1=非增强核心、2=周围非增强异常、3=增强组织、4=切除腔；全肿瘤WT（whole tumor）=[1,2,3]，肿瘤核心TC（tumor core）=[1,3]。
nnU-Net官方 https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/documentation/region_based_training.md 说明重叠区域编码在部分任务优于单独类别，且其示例重编码后的1/2含义与本项目不同，不能照抄TC=[2,3]。本次仅保留原始类别以支持后续严格对照，不声称五分类一定比二分类好。旧RC独立辅助头0.1失败，没有检验完整前景区域编码，但本工程也不重开那个失败配方。

## 4. 具体怎么做？

只读fit684原生整数seg与已知源SHA；沿用旧image-only裁剪框，最近邻精确采样到原128³，保存0—4整数类。可在读取时派生WT=[1,2,3]、TC=[1,3]、增强组织ET（enhancing tissue）=3、切除腔RC（resection cavity）=4。WT必须逐体素等于旧WT cache，RC也必须逐体素等于旧辅助cache；不改变已有图像或标签。
独立验证用NumPy中心坐标索引直接从原生标签取样，与构建阶段Torch nearest-exact比较所有684例，不只重算自身报告。没有模型/教师/loss/新预测输出。

## 5. 唯一变化与固定项

仅新增一份保留类别信息的数据视图；不修改原训练标签、data loader、模型、采样、损失、168轮或任何当前门。病例固定fit684/313、原source/hash/crop/128栅格/类别语义不变，不读dev/test标签。当前模糊训练明确不接入这份cache。保留的是现有128栅格能表达的类别，不能恢复已被缩小丢掉的原生细节。

## 6. 成功、失败和停止规则

源SHA及AUX来源收据、case顺序/fitSHA、原生shape/RAS方向、整数0—4、旧WT与RC byte parity全部通过；独立索引重建全684类数组byte exact；区域嵌套WT>=TC>=ET、WT与RC不重叠。任何源/几何/映射不一致立即停、保留部分访问，不排病例/重标注或调取样规则。工程成功不准入模型训练，需要未来另定单因素目标编码协议与对照。没有Dice/Point提升门、bootstrap或test准入。

## 7. 成本与风险

uint8 684×128³=1.336GiB，/tmp优先；4 CPU workers，预计5—20分钟但远端读取可能更慢，0 GPU/模型前向，不抢当前模糊GPU。原始镜像与官方字节parity仍未证明。更多标注信息不保证更好模型，改变标签编码常会连带输出/损失，需要未来明确控制，不能把本cache当已验证的新分割方法。

## 8. 最终结果（完成后填写）

尚未构建/核验；当前仅工程预注册，当前正式训练仍只有图像模糊双seed。
