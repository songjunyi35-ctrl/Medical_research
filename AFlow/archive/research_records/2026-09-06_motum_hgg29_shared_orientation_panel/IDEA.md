# Experiment idea: MOTUM-HGG29共享方向冻结面板

Status: planned; shared-orientation input bridge passed; MOTUM performance remains unaccessed.

## 1. 现在的问题是什么？

MOTUM29旧性能实验在0模型时因两个mask各自canonical affine与image相差`1.4488e-5`而关闭。随后独立input-only桥证明：29/29 native shape一致、affine≤`2.384e-7`、离散orientation相同；从image计算一个共享轴变换时，两个独立进程的三array/shared-affine SHA完全一致且转换arrays与旧canonical arrays bit-exact。MOTUM性能仍未访问。

## 2. 一句话直觉

题目和答案纸不变，只让三张已经对齐的透明胶片共用一把坐标尺，再让冻结模型参加一次厚层MRI小考。

## 3. 为什么选择这个idea？

- Bridge结果SHA `3d349094…`直接解决旧实现的header浮点放大，不移动或插值GT。
- 旧实验失败在N4/cache/checkpoint之前，只否定独立canonical affine门，未否定共享离散方向接口或冻结checkpoint。
- HGG29仍是官方双传输exact、三家中国医院、医师校正双区域标签的未见cohort；38例转移瘤在性能前排除。
- 当前GCSAM五cohort只有Africa CI全正；Flip8四cohort点估计全正、三处CI全正。MOTUM可作新方向检验，但29例预期CI宽。
- 一手依据：MOTUM [官方DOI](https://doi.gin.g-node.org/10.12751/g-node.tvzqc5)；离散方向接口为[nibabel官方orientations API](https://nipy.org/nibabel/reference/nibabel.orientations.html)。

## 4. 具体怎么做？

读取T1ce和两个语义mask，先验证native shape exact、affine≤`5e-7`与orientation exact；从image计算一次到RAS的flip/permutation，同样应用到三arrays，共享image-derived affine，0插值。WT为两个mask正类并集。随后canonical N4 shrink2、image-positive bbox、positive z-score、image trilinear/label nearest到128³。

Full KD168运行identity+固定八镜像raw-logit均值，GCSAM184 identity；阈值0、Point=argmax、无SAM/后处理/模型ensemble。29 scans=subjects，报告3绝对行、2配对差及10,000 bootstrap。

## 5. 唯一变化与固定项

- 相对旧失败实验唯一变化：各文件独立canonical affine→通过bridge的image-derived shared orientation/affine。
- 固定：HGG29顺序、38转移瘤排除、source arrays、WT双mask正类并集、N4/crop/z-score/128³、checkpoint、view、阈值0、无SAM/后处理、bootstrap与解释门。

## 6. 成功、失败和停止规则

- 工程：bridge结果SHA固定；每次prepare重验native≤`5e-7`/orientation exact、shared arrays与各自canonical arrays exact、0 WT crop loss；首例/全cache byte exact；smoke/preaccess全哈希。
- 性能：GCSAM和Flip8各按delta>0且CI lower>0=`supported_on_MOTUM`、delta>0且跨0=`positive_but_uncertain`、delta≤0=`not_supported_or_reversed`。
- 无promotion；任何结果完整发布并关闭，不重跑/调参/重访旧cohort。

## 7. 成本与风险

- 训练0；cache约290 MiB、CPU数分钟；9 synthetic+261 formal forwards，峰值预计5.1 GiB。
- Full Flip8部署8×，identity 1×；参数不增。
- 21/24 slices、5.5–6.75mm厚层、半自动后医师校正、positive-support N4、N=29、无patient bridge和checkpoint训练源parity限制不变。

## 8. 最终结果（完成后填写）

- External：未运行；首例0-WT-crop-loss门失败。
- CI/病例：无性能。0001 raw WT 21,007，positive-image bbox内20,115，丢892/4.246%；全29仅这一例丢失。
- 直觉：共享坐标直觉支持，但没有解决processed负强度与positive-support crop冲突。
- 成本/访问：0 cache/checkpoint/GPU/forward/performance；只读29例诊断。
- 下一步：当前协议关闭；不排例、不用GT扩框、不自创offset救场。
