# 训练期切除腔辅助监督：冻结协议

2026-09-07，implementation/training/dev性能前冻结。仅限用户批准reused-development，非fresh/blind。

- Split：`archive/research_records/2026-09-07_reused_development_roles/SPLIT.json`；fit684/313 SHA=`75504df75fa734723945a9f07d76696fdcca89b8c29563dc0ad02a2560718d61`；dev180/78 SHA=`53885b7e300f915d81eecb115b82123e25330b7da422d3ff0b889b0489a2cdac`。全部保护旧confirmation53、fixed validation50、test119患者；不增加外部病例。
- Input：已有canonical N4 braincrop128 train972 cache中只按fit/dev ID索引；index SHA=`7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`，WT SHA=`bf470d48a5f115f09407ad351bb6dee11ee36bdd0c2a4285ae0bd9cc22125724`。Formal前重hash影像/WT并绑定预访问收据。
- Auxiliary cache：仅fit684 raw seg，经nibabel closest canonical、旧image-only crop geometry、nearest-exact128重采样得到RC=(raw==4)。同样重建WT=(raw1|2|3)必须与旧cache逐例byte exact；RC与WT不可交叉。图像与dev RC不读/不重建。逐源SHA与RC缓存SHA保存。
- Model：MONAI SegResNet in1/out2，width16、down(1,2,2,4)、up(1,1,1)、dropout0.1；两臂完全同结构/初始化。主bias0、辅bias−4。最后裁去辅助输出，主logit保持。没有历史checkpoint、没有teacher/KD、无EMA。
- Loss：WT沿用DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1)。设m=1−WT，q=sigmoid(aux)，r=RC；每例辅助Dice loss为1−(2sum(m*q*r)+1)/(sum(m*q)+sum(m*r)+1)；辅助BCE为sum(m*BCEWithLogits(aux,r))/max(sum(m),1)，batch平均。候选L=L_WT+0.1*(L_auxDice+0.1*L_auxBCE)；control相同计算但外层系数0。固定无pos_weight、无边界加权、无额外support筛选。
- Pair seeds：2026090711、2026090712。每seed control与candidate相同初值与RNG。先pair11再pair12，每对两卡同时启动；第二pair不因第一pair训练loss或dev结果挑选。四条完整训练冻结之前不能计算dev性能。
- Training：168 epochs固定；每轮972 replacement draws、batch8/drop_last→968有效draw/121步；native WT inverse-sqrt采样仅fit684，clip[.5,4]；AdamW lr2e−4/wd1e−5、100轮恒定后用既有200轮cosine schedule到第168轮、minLR1e−5；BF16；clip grad norm5；固定max_shift8、三轴随机翻转、scale[.9,1.1]、offset[−.1,.1]、noise[0,.05]。增强将WT/RC拼通道共同变换。num_workers2；Torch/SimpleITK单线程。每臂墙钟6h和allocated24GiB安全门。
- Evaluation：四checkpoint全部完成后dev180各identity一次，batch8、BF16、WT阈值0、point=WT raw-logit argmax；无TTA、SAM、连通域或其他后处理。主要Subject Dice（患者内先平均），次要Scan Dice及Point；不看RC作为选择指标。Dev total720单例forwards。
- Bootstrap：10000次percentile，seed base2026090730；对pair11/12分别base+0/+100。同一次患者重采样用于Subject等权差和Scan加权差，保留每个患者全部扫描；CI用于开发描述/门而非独立test显著性。每模型绝对CI用base+1000+模型序号。
- Gates：每seed candidate−control Subject≥.003且Scan>0、Point≥−1，Subject与患者聚类Scan CI下界均≥−.003；两seed平均Scan≥.003；所有工程门通过。否则关闭当前配方，不搜weight/epoch/seeds；全部通过仅允许称reused development supported，不自动访问其他cohort。
- Output：四条TRAIN_DONE和fixed checkpoint、source/cache receipts、IMPLEMENTATION_FREEZE、SMOKE_RECEIPT、TRAIN_PREACCESS、DEV_PREACCESS、RAW_DEVELOPMENT、RESULT.md/JSON；本实验只更新方法目录与handoff，开发结果不进入test榜。
- Provenance：训练源为本地BraTS2024 Kaggle mirror，官方Synapse parity仍未证明。用户仅授权复用开发，未授权外部数据协议签署或旧test调参。
