# Experiment idea: single-bottleneck DMRFB dilation adapter

Status: completed fixed-validation partial mechanism support; promotion failed without test access

## 1. 现在的问题是什么？

当前最佳单模型KD200 fixed validation Scan/Subject/Point=`0.734384308/0.724434143/0.935185`。LoMix、DCD、KAS、hACE、VM Dice和Batch-CVaR50的训练目标都实际学动，但都没有提高overall；这说明继续修改loss或teacher的成功概率很低。

错误分层显示高边界/碎片化病例仍较难。局部patch可行性又发现固定96³原生ROI只有74/108病例覆盖至少90% GT，最差仅15.9%；NMSW在当前单肿瘤/128³输入还会退化。因此下一步只改变最深特征的多尺度感受野，不增加第二模型、patch或loss。

## 2. 一句话直觉

在模型最深处给它三副不同焦距的眼镜，让同一个位置同时看近邻、中距离和更远上下文，但保留KD200已经学好的主干。

## 3. 为什么选择这个idea？

- MICCAI 2025 EfficientMedNeXt提出Dilated Multi-Receptive Field Block（DMRFB）。3D BTCV单因素消融从Dice`85.58→86.06`，约`+0.48`；完整模型在MSD BrainTumour比MedNeXt-L_K5约高`0.40`。
- 它作用于网络表达能力而非loss，前述负结果没有测试这一机制。
- 不能直接替换KD200的trained ResBlock，因为dilation/MedNeXt结构无法保持函数。采用最深encoder后的residual adapter，可以zero-output初始化并严格保留KD200初始logit。
- 为排除“只是多了57k参数”，两臂都加同参数adapter，唯一差异是第三分支的dilation/padding。
- 来源没有BraTS2024、T1ce-only/WT或single-bottleneck证据；项目是source-inspired机制实验，不是完整EfficientMedNeXt复现。

Primary sources:

- Paper: <https://papers.miccai.org/miccai-2025/paper/4895_paper.pdf>
- Official code: <https://github.com/SLDGroup/EfficientMedNeXt>
- Frozen official commit: `803f7efed9b728ac93ae4e0d8a2602501135241f`

## 4. 具体怎么做？

两张A100各训练一臂，都从KD200开始，无teacher。

原SegResNet encoder输出最深特征`x:[B,128,16,16,16]`。它进入训练后保留的adapter：

1. 三条depthwise Conv3d并行；
2. 输出concat成384 channels；
3. `GroupNorm(num_groups=128,num_channels=384,eps=1e-5,affine=true)`；
4. GELU；
5. `Conv3d(384,128,kernel1,bias=true)`投影；
6. 加回identity `x`，再走原decoder和final head。

共同branch1=`k1,d1,p0`，branch2=`k3,d1,p1`。

- Control：branch3=`k3,d1,p1`，有效感受野`[1,3,3]`。
- Candidate：branch3=`k3,d2,p2`，官方DMRFB有效感受野`[1,3,5]`。

两臂所有参数shape、tensor初始化、norm、activation和projection完全相同；仅branch3的dilation/padding metadata不同。最终projection weight/bias严格初始化0，所以epoch0两臂和plain KD200 final logits逐元素相同。

Loss仍是历史DiceCE；输出仍为raw final logit阈值0。

## 5. 唯一改变什么，哪些保持不变？

唯一因果变化：adapter branch3从dilation1/padding1变为dilation2/padding2。

固定：

- KD200起点SHA `56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7`；历史validation/test访问如实披露。
- 两臂相同57,088 adapter参数；总部署参数4,757,985。
- adapter construction seed20260901；training seed20260900。
- train756/346、true128 cache、batch8、weighted replacement、shift8。
- base parameter group：fresh AdamW LR`1e-5`、weight decay`1e-5`。
- adapter parameter group：同一AdamW，LR`1e-3`、weight decay`0.08`，采用source optimizer scale使随机新模块能在32轮内学习；两臂完全相同。
- betas`0.9/0.999`、eps`1e-8`、clip5 over all parameters。
- global201–232、save208/216/224/232；same case/RNG pairing。
- validation108、Dice/Subject/Point、threshold0、无SAM/TTA/postprocessing。
- 部署保留adapter，单模型、一次forward。

Source从头训练96³、五头deep supervision、AdamW LR`1e-3`/wd`0.08`、BrainTumour 60k steps；项目是KD200后32轮双LR continuation，明确外推。

## 6. 成功、失败和停止规则

Selection只按candidate在208/216/224/232中的Scan Dice最大，Subject exact tie，最终同分选更早epoch。

Promotion全部必须通过：

1. selected candidate Scan/Subject严格超过same-epoch adapter control；
2. selected candidate Scan/Subject严格超过KD200；
3. Point至少101/108；
4. 相对control和KD200的paired Scan/Subject CI下界均`>=-0.003`；
5. boundary-top27 candidate−control mean Dice严格`>0`；
6. KD200-bottom27 candidate−control mean Dice`>=0`；
7. implementation、initial identity、two-step gradients、pairing、reload、memory和access门全部通过。

Technical stop：两臂adapter tensor初值不同；projection非全0；epoch0 plain/control/candidate logits最大差非0；step1 projection无finite nonzero gradient；第一次optimizer update后的step2任一branch/norm/projection无finite nonzero gradient；参数遗漏optimizer；batch8 peak>24GiB；外推>45min；checkpoint reload/deploy不等价。

失败后不搜索dilation、branch数、插入层、adapter LR/WD、epoch、threshold或postprocessing；不访问test270。

## 7. 代价

- 两臂双卡并行，预计20–25min墙钟。
- 每臂新增57,088参数，总4,757,985，约+1.2%。
- 预计peak<20.8GiB，24GiB硬门。
- 无新cache；8个checkpoint约150MB。
- 推理仍1×model forward，预计延迟增加1–5%，无teacher或第二模型。

## 8. 最终结果（完成后填写）

- Validation：candidate按Scan选择epoch216，Scan/Subject/Point=`0.734242477/0.724742362/0.925926`；same-epoch same-parameter control=`0.733997313/0.724596003/0.935185`；KD200=`0.734384308/0.724434143/0.935185`。
- Test：promotion失败，未授权、未访问。
- Boundary/bottom-tail：candidate-control boundary-top27=`+0.000107092`，bottom27=`+0.001184257`；两机制门通过但CI均跨0。
- Cost：双臂formal约18.58/18.55min，双卡墙钟18.58min；peak allocated约20.40GiB；每模型4,757,985参数、1×forward。
- 是否支持直觉：部分支持。candidate超过同参数control的Scan/Subject并改善两个hard strata，说明dilation2方向有小收益；但Scan仍比KD200低`0.000141831`且Point少1例，不满足部署/test门。
- 下一步：当前single-block协议停止，不搜索dilation/block/LR/epoch。可独立预审“KD200 zero-adapter与DMRFB216同架构同盆地weight interpolation”作为单模型折中，但必须重新预注册且不得用test选alpha。
