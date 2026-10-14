# Experiment idea: Full972 保统计Gamma强度增强

Status: planned and frozen before split generation, implementation, GPU model construction, training, or new evaluation access.

## 链路审视与启发

当前链路是：`raw NIfTI → subject角色冻结 → N4 + image-only brain crop + true-128³ + 每病例非零区域z-score → online augmentation → random-init Full972 voxel-KD → train-only inner → fresh Full972 final → fixed validation gate → gated reused test report`。

1. Image-only brain crop约`+0.054` Scan、epoch99→168约`+0.0154`、972例从随机初始化参与相对matched756为`+0.010943/+0.007982`，说明数据呈现与早期信息比堆容量/模块更可靠。
2. 当前online增强已有三轴翻转、±8体素平移、每病例线性scale`0.9–1.1`、offset`-0.1–0.1`和高斯噪声`0–0.05`，但没有非线性强度曲线。
3. LAWA6在三个seed都小幅提高自身raw，却final仍低于archived强seed`0.007093/0.004897`；Lookahead5虽降低fit loss，fresh inner却`-0.003897/-0.004757`。优化器/权重平均轴不能稳定复制强seed。
4. T1ce MRI虽已每病例z-score，但不同扫描仪、采集和造影反应仍可能改变组织间相对对比曲线；线性scale/offset不能模拟中间灰度的非线性压缩/拉伸。
5. 旧intensity residual在预测后用灰度修mask，导致Scan/Subject`-0.002638/-0.003331`且Point大降；这不等于训练期让同一标签适应轻度非线性对比变化。
6. Fixed validation与test已多次揭示，本实验属于development/post-test research，不使用test逐病例结果选Gamma设置。

采用的启发是第1、2、4条：离开optimizer/loss/teacher/容量轴，补足输入增强中一个明确缺失的MRI强度不变性。

## 1. 现在的问题是什么？

Archived Full972 seed32 fixed validation为`0.740013/0.731266/103/108`，而最近fresh final/inner随seed明显变化。当前每病例z-score和线性亮度增强无法模拟“暗区、中灰区、亮区被不同程度拉伸”的非线性扫描对比。模型可能依赖特定T1ce亮度曲线，造成跨病例/seed泛化不稳。

## 2. 一句话直觉

像给同一张MRI调照片曲线：保持明暗顺序与整体均值/波动，但随机压暗或提亮中间灰度，让模型认肿瘤形状和上下文，而不是背一种固定对比度。

## 3. 为什么选择这个idea？

- nnU-Net是跨23个医学分割数据集验证的自配置框架，其默认训练增强包含Gamma correction。
- 官方nnU-Net代码使用非反相Gamma范围`0.7–1.5`、retain_stats=true、per-sample probability0.3；官方batchgenerators提供3D兼容GammaTransform与retain_stats定义。
- 项目当前增强已覆盖翻转、平移、线性亮度与噪声，Gamma是清晰的单一缺口。
- Intensity residual负结果改变推理输出并制造GT外高峰；本方法只改变训练输入、保持标签和部署完全不变。
- 160³/width24/LoMix/DCD/校准/困难loss/多seed/optimizer负结果均未测试保统计的非线性输入增强。

Primary sources:

- nnU-Net paper: https://www.nature.com/articles/s41592-020-01008-z
- Official nnU-Net repository: https://github.com/MIC-DKFZ/nnUNet
- Official batchgenerators GammaTransform: https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/color_transforms.py
- Official batchgenerators framework: https://github.com/MIC-DKFZ/batchgenerators

来源边界：本项目只适配单个Gamma增强，不复制nnU-Net架构、patch sampling或完整增强栈，不能称nnU-Net复现。

## 4. 具体怎么做？

### Gamma操作

两臂先执行现有完全相同的flip/shift/scale/offset/Gaussian noise。Candidate用独立CUDA Generator逐病例以概率`0.3`决定是否加Gamma，避免改变原增强随机流。

对选中病例的单通道128³ tensor：记录原均值`mu`、标准差`sigma`、最小值`lo`和最大值`hi`；若`hi-lo<=1e-6`则拒绝formal。将强度归一到`u=(x-lo)/(hi-lo)`。Gamma范围跨1时遵官方语义，先以0.5概率从`[0.7,1]`或`[1,1.5]`均匀取`g`；计算`y=u^g×(hi-lo)+lo`；最后`y=(y-mean(y))/std(y)×sigma+mu`，保留原均值和标准差。标签不变。

### Stage A

新建Full972 development-train的subject-clean`864 fit/108 inner`。Control/Candidate使用同一seed20261081、exact初始state、sampler和现有增强；唯一差异是Candidate在现有增强后添加Gamma。固定训练到epoch168且训练器不构造inner loader。配对工件全过后才一次性评fresh inner。

### Stage B

只有Stage A全过门才用fresh seed20261082在full972做配对final到epoch168，冻结后一次性访问fixed validation108。

输入→teacher→loss→输出：增强后的T1ce进入同一4.70M student和固定history/EMA teacher；真实WT DiceCE加`0.3×T²`逐体素KL更新student；最终输出仍raw logit，阈值0，最高logit为自动point。Gamma和teacher都不进入部署。

## 5. 唯一变化与固定项

- 唯一变化：Candidate在现有增强之后，以独立RNG添加`p=0.3, gamma=[0.7,1.5], invert=false, retain_stats=true`的单通道Gamma；Control不添加。
- 固定：数据/角色、brain crop、true-128³、z-score、原flip/shift/scale/offset/noise增强及其global RNG、4,700,897参数SegResNet、dropout0.1、两臂相同seed/初值、native-volume sampler、batch8、972 requested/968 consumed/121步、teacher、DiceCE+voxel-KL、AdamW、late-cosine、epoch168、clip5、阈值0、无TTA/SAM/postprocess、1×推理。
- Stage A Gamma Generator seed20261181；Stage B seed20261182。Gamma RNG不得改变global CPU/CUDA RNG。

## 6. 成功、失败和停止规则

### Stage A

- Candidate必须严格超过Control Scan/Subject，Point不得下降。
- Scan/Subject paired bootstrap 95% CI下界均`>=-0.003`；10,000次，seed20261083。
- 两臂初值、168轮sampler SHA、首批原增强pre-Gamma image/label SHA exact；Candidate Gamma应用比例在`[0.25,0.35]`，所有g在`[0.7,1.5]`且非1，应用样本均值/总体标准差误差<=`1e-4`，label exact，Gamma RNG不改变global RNG。
- 任一失败不运行Stage B。

### Stage B

- Candidate必须严格超过fresh matched Control和archived Full972 seed32`0.740012535/0.731266217`的Scan/Subject，Point>=103且不低于Control。
- 四个paired CI下界均`>=-0.003`；10,000次，seed20261084。
- 全过门才允许一次冻结的reused test270报告；否则不访问test。

### 工程停止

- SHA/access/leakage/pairing/Gamma概率/范围/统计/RNG/label/reload/deploy/finite不符；teacher有梯度；peak>24GiB；单run>6h，立即停止。
- 失败后不调Gamma概率/范围/采样分布/顺序/retain_stats、seed、epoch、LR、loss、threshold、TTA、SAM或postprocess。

## 7. 成本与风险

- 每阶段配对双卡约3.5–4h；两阶段均运行时墙钟约7–8h、formal约14–16 GPU-hours。
- Gamma逐病例做min/max/mean/std/pow，预计小幅训练耗时/显存增量；24GiB硬门。
- 不建cache；workspace预计<300MB。
- 推理仍4.70M参数、1×forward，无部署变化。
- 最大风险：现有z-score+scale/offset/noise已足够，Gamma冗余；或对全volume含噪背景做非线性变换损伤边界/Point。

## 8. 最终结果（完成后填写）

- Stage A：待运行。
- Stage B：仅Stage A全过门后运行。
- Test：仅Stage B全过门后一次固定报告。
- CI/病例分布：待运行。
- 直觉支持：待运行。
- 证据/哈希：待运行。
