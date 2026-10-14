# 实验idea：T1ce脑区固定1–99百分位裁剪稳定性

状态：七问与链路审视已冻结；尚未实现、建cache、构造新split、训练或访问新inner

日期：2026-09-03 UTC

## 链路审视与启发

当前完整链路是：

`BraTS raw NIfTI → 按患者划分manifest → canonical RAS/N4/image-only brain crop/非零z-score/128³ cache → 4.70M SegResNet + history/EMA teacher + DiceCE/voxel-KL → train-only inner冻结轮次/门槛 → fresh-seed Full972重训 → fixed validation gate → reused test270或独立外部报告`。

每一环的证据与瓶颈如下：

1. **Raw data与角色：** 972 scans/444 subjects只属于train角色，可以继续生成新的subject-disjoint train-only inner；fixed validation108、outer、confirmation、reused test270与UCSD75都已揭示，不能再为新方法收尾或选参。RHUH人工标签外部队列尚无官方可验证inventory，不能用第三方镜像替代。
2. **Cache与输入表示：** image-only brain crop在旧匹配协议中约提高Scan `+0.054`，是最大可靠单项收益；true-160³让体素计算约`1.95×`却把test Scan降到`0.737060`，说明“去无效背景”有用，“无差别加像素”无用。当前N4后直接用非零总体均值/标准差做z-score，尚未隔离测试极端强度值是否影响每例亮度标尺。
3. **训练时机与数据利用：** Full972从随机初始化第一轮纳入新增216例，相对matched Full756 fixed validation提高Scan/Subject `+0.010943/+0.007982`；只在后32轮增加draw budget的收益仅`+0.000779/+0.000954`。真正有用的是早期表示学习看见更多病例，不是简单多走几步。
4. **Teacher与loss：** 固定logit ensemble和voxel-logit knowledge distillation（逐体素logit知识蒸馏）建立了最强单模主线；ReCo、DCD、KAS、难例CVaR、VM Dice等更复杂监督没有稳定超过它。当前不再改teacher、loss或困难病例权重。
5. **模型容量：** width24相对width16在inner曾提高Scan/Subject `+0.010634/+0.012193`，但fresh final反向`-0.002872/-0.003525`；更大容量没有解决泛化。新方法保持4.70M width16。
6. **稳定性与指标分叉：** Gamma在fresh inner为`+0.006151/+0.005570/+1 Point`，换fresh Full972 seed后变成`-0.003848/-0.002312/0`；GCSAM固定184双复制平均Scan为`-0.000519`。三seed ensemble虽提高Scan `+0.005829`，Subject CI下界仍为`-0.005660`。Point又常与Dice分叉，因此不能靠一个seed、一个点估计或单一指标晋级。
7. **证据强度：** internal test最高Scan点估计GCSAM `0.752003`与Full972 KD `0.751760`只差`0.000244`且CI跨0；UCSD75的GCSAM优势也由少数大救援驱动。它们只能描述历史，不能决定本实验的百分位、epoch、阈值或病例。

由此形成五条数字化启发：

- **启发一：优先改输入信息质量，而不是再堆容量。** brain crop约`+0.054`支持表示杠杆；160³和width24负结果限制了“更多像素/参数”解释。还缺的是在同一T1ce任务内对强度表示的配对证据。
- **启发二：方法必须跨两个fresh seed同时成立。** Gamma从`+0.006151`翻到`-0.003848`、GCSAM双复制平均Scan `-0.000519`支持这一点；限制是同一972病例池仍不等于独立医院。还缺未来人工标签外部cohort。
- **启发三：972例要从训练起点参与。** Full972相对Full756约`+0.010943`，而后期加draw只约`+0.000779`；限制是现有fixed validation已揭示。新实验Stage A用864 fit作干净筛选，只有双复制通过才在全部972上从随机初始化训练冻结候选。
- **启发四：保留成熟teacher/loss，只隔离输入变量。** Full972 KD是fixed validation `0.740013/0.731266/103`和internal test Subject/Point最高单模；复杂loss/teacher routing多次失败。还缺确定性强度去极值是否让同一teacher-student链路更稳。
- **启发五：Scan、Subject与Point必须共同设门。** GCSAM复制A提高Subject/Point但降低Scan，B略升Scan却降低Subject并少2 Point；所以三项在结果前冻结，禁止结果后挑指标。

## 1. 现在的问题是什么？

当前最强可比较单模Full972 KD在fixed validation108的Scan/Subject/Point为`0.740013/0.731266/103/108`。看似更高的GCSAM epoch184在reused test270为`0.752003`，但双新seed复制的平均Scan差为`-0.000519`，没有稳定优势。优化器、容量、160³、随机Gamma和复杂loss轴均未稳定提高；尚未隔离的问题是：N4后的少量极亮/极暗脑内体素是否牵动每例z-score，使同一组织在不同扫描中尺度不稳。

## 2. 一句话直觉

像拍照前先剪掉最刺眼和最黑的各1%坏点，再统一亮度；不要让少数极端像素决定整张MRI的亮度尺子。

## 3. 为什么选择这个idea？

- 最大可靠收益来自image-only brain crop，而不是增加参数/分辨率，说明输入表示值得优先处理。
- Crisan与Borza 2026术后胶质瘤预印本的Norm arm使用脑mask内1–99百分位裁剪后z-score，并报告跨域强度/损失稳定性问题；仓库已完成一手适配审计。论文Norm arm同时加入对比学习，不能把整包结果归因于裁剪，所以这里只隔离一个因素，不称论文复现。
- Gamma负结果没有否定本方法：Gamma以约30%概率随机扭曲整条亮度曲线；这里每例确定性删除两端极值，目的和算子都不同。
- 160³、width24、GCSAM、LAWA6、Lookahead和多seed routing都没有改变N4后z-score的极值敏感性，因此未预先否定本实验。
- 论文在线地址`https://arxiv.org/abs/2607.22749`在本session的联网索引返回404；本实验只依赖已归档的[一手论文适配审计](../../reviews/2026-09-02_loss_stabilization_paper_fit_audit/RESULT.md)，不虚构新的在线核验。

## 4. 具体怎么做？

Control输入路径保持：raw T1ce → canonical RAS → N4 bias-field correction（偏置场校正）→ 按图像非零区裁脑 → 非零总体z-score → trilinear resize至128³。

Candidate只在N4与z-score之间增加：对每例裁脑后的非零体素计算固定第1与第99百分位，把低于/高于阈值的值夹到阈值，再用裁剪后非零总体均值和总体标准差做z-score，最后以相同方式缩至128³。标签只做相同nearest-exact resize，必须与Control逐字节一致。

128³图像进入同一4.70M SegResNet。冻结history168与EMA168老师在各自arm的输入上产生logit，student仍优化Dice加0.1交叉熵以及权重0.3、温度2的逐体素Kullback–Leibler divergence。输出仍是raw logit `>0` mask，最高logit是自动point。

## 5. 唯一变化与固定项

- **唯一变化：** N4后、source非零总体z-score前是否执行每例固定1st/99th percentile clip。
- **固定：** T1ce/WT=`{1,2,3}`、两个fresh subject-disjoint split、每个pair的seed/初始state、fit病例、原生WT体积sampling、每轮972 requested/968 consumed/121 steps、增强、4.70M width16、history/EMA teacher、DiceCE+voxel-KL、AdamW、batch8、late-cosine、固定epoch168、阈值0、无TTA/Segment Anything Model/后处理、部署1×。
- Candidate cache放在`/tmp`的独立大容量盘；不删除仓库材料，不创建仓库多GiB cache。

## 6. 成功、失败和停止规则

- **Primary：** 两个新train-only inner108各自的Scan Dice；共同安全指标为Subject Dice与Point hit。
- **Stage A：** 两套新`864 fit/108 inner`复制；四条Control/Candidate训练全部到固定epoch168并冻结检查点后，才由独立evaluator对两个inner各访问一次。
- **每个复制门：** Candidate−Control Scan>0、Subject>0、Point hits≥−1；Scan case-paired与Subject subject-paired 10,000次bootstrap 95% CI下界均≥`-0.003`；初始state、sampling weights、168轮sampler、增强参数/标签及工程门exact。
- **跨复制门：** 两套Scan delta均值>0、Subject delta均值>0。
- **失败：** 任一门失败即停止，不加第三个seed，不调百分位/顺序/epoch/阈值，不训练Full972。
- **成功：** 只授权fresh seed的Full972 Control/Candidate从随机初始化训练至固定epoch168并冻结候选；不访问fixed validation、outer、confirmation、test270或UCSD75。未来只能在新的人工标签cohort上确认。

## 7. 成本与风险

- Stage A四条训练预计约13.6 GPU-hours；两卡并行Control与Candidate，墙钟约6.8h。
- 若全过，Full972 Stage B两条再约6.8 GPU-hours、3.4h墙钟。
- 峰值显存预计每卡约20.3GiB，硬门24GiB。
- Candidate float32 image+label cache约9.5GB，写入`/tmp/aflow_t1ce_percentile_clip_20260903`；`/tmp`当前554GB可用。仓库仅新增脚本、报告与约<300MB检查点。
- 推理仍1×locator forward，只多固定分位裁剪与z-score预处理；模型参数和部署图不变。
- 最大技术风险：N4后极值可能本来就是有用病灶信号；clipping也可能使冻结teacher在新输入上退化。双新seed门会直接检验而不搜索设置。

## 8. 最终结果

Stage A两个fresh train-only复制均失败。A Candidate−Control Scan/Subject/Point=`-0.008210/-0.009950/-4`，B=`-0.007520/-0.009875/+5`；四个Dice paired 95% CI均整体低于0，两复制平均Scan/Subject=`-0.007865/-0.009912`。全部cache、训练与配对工程门通过，故直觉不支持且`stage_b_authorized=false`。未训练Full972，未访问fixed validation/outer/confirmation/test270/UCSD75。完整结果见[RESULT.md](RESULT.md)与[RESULT.json](RESULT.json)。
