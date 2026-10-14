# 灰度处理的公开代码核对（2026-09-17）

本材料仅做论文与代码阅读，未训练、未加载模型、未新增验证或测试推理。源地址与本次抓取 SHA 见 [抓取收据](PUBLIC_CODE_SOURCES.json)。本材料不能替代训练链、数据与评估审计。

## 结论

当前实现不能称师兄整篇 MSCL 的完整复现，也无法称逐行复现作者灰度模块。它是匿名稿公式 5 的一个明确但尚未经作者确认的可微实现。这里“未完整复现”不等于已经发现算错公式；更不等于能把掉点全部归因于灰度项。

## 原作者代码找到了吗

本地原稿题名为 **MSCL: Multi-stage Curriculum Learning for Postoperative Glioma Segmentation Based on Fine-tuning Foundation Model**；作者栏为 Anonymized Authors，没有作者姓名、机构、代码 URL 或可确定的发布标识。

已检索：完整题名；“MSCL”+“glioma”；“Multi-stage Curriculum”+“postoperative glioma”；“gray-scale threshold loss”；以及 GitHub、arXiv、OpenReview 限域同义检索。没有查到可核对为本稿原作者的公开仓库或完整公开论文。检索结果中的其他 MSCL（生物离子通道或其他分割论文）不匹配。准确表述是“本次检索未找到且作者身份仍匿名”，不能断言全网不存在代码。

原稿 [本地文本](../../research_records/2026-09-16_sammed3d_graymean_single972/MSCL_SOURCE.txt) 第 2.3 节：分割项是 Dice 与交叉熵各半；灰度项是初始提示下预测肿瘤区域和真实肿瘤区域的平均灰度之均方误差。第 3.1 节给出 batch 1、学习率 8e-4、weight decay 0.1、200 epochs、128³；声称有归一化、强度校正与去颅骨，但没有逐步参数。

它没有明确：预测区域软/硬定义、灰度单位及计算发生在归一化之前还是之后、灰度总权重、批内归约、空区域处理、数值稳定项、是否只对第一次输出加一次灰度项。初始 p0 的文字支持“首提示”解释，但不足以证明作者具体程序如此实现。文中还包含完整三阶段课程与残余肿瘤模拟，本轮单组灰度训练没有实现整套课程，这是已知范围而非隐蔽错误。

## 真正读到的公开实现

| 代码及准确匹配度 | 实际处理 | 对本项目的意义 |
|---|---|---|
| [SAM-Med3D 官方 train.py](https://github.com/uni-medical/SAM-Med3D/blob/main/train.py)，骨干训练来源，非 MSCL 灰度实现 | 真实标签中心裁剪；用正值体素计算 Z 标准化；DiceCELoss 使用 sigmoid 与 squared_pred；11 次交互；随机一次和最后一次不提供点；编码器/提示和解码器学习率为 8e-4/8e-5 | 官方没有区域灰度约束。当前本地首提示灰度项是新增内容，不能以官方代码证明该项正确复现作者 |
| [SAM-Med3D 点击代码](https://github.com/uni-medical/SAM-Med3D/blob/main/utils/click_method.py)，提示采样来源 | get_next_click3D_torch_2 比较预测 logit 与 0.5，再从真假标签差异区域采点 | 这里 0.5 是预测输出阈值，不是 MRI 灰度阈值。train.py 的 threshold=1000 是前景体素数筛选，也不是灰度阈值 |
| [Active-Contour-Loss 作者仓库](https://github.com/xuuuuuuchen/Active-Contour-Loss/blob/master/Active-Contour-Loss.py)，相关思想，非等价实现 | 函数输入是 y_true/y_pred；区域项以真实二值标签和常量 1、0 构造，另有边界长度项 | 不能复制这段代码后声称得到师兄的 MRI 灰度均值约束。其区域项没有拿 MRI 输入作两区域均值差 |
| [scikit-image v0.25.2 Chan–Vese 源码](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/segmentation/_chan_vese.py)，真实图像区域均值方法，非 MSCL | 先把图像缩放至 0–1；计算轮廓内外加权均值，再用像素与两均值的平方差及边界长度演化轮廓。最终 phi>0 得到区域 | 它优化轮廓方程，不是 SAM 网络反传。直接把硬区域均值搬进神经网络会失去常规梯度路径；不能用“别人用了硬区域”证明此处也该硬阈值 |

必须补充：官方工具文件另有 `get_next_click3D_torch_no_gt(..., threshold=170)`，这里确实是图像强度阈值；详见下方调用链核对。上表只解释 1000 和 0.5，不能概括成官方没有任何灰度阈值代码。

源代码定位：SAM train.py 的 BaseTrainer.set_loss_fn/set_optimizer/interaction；点击文件的 get_next_click3D_torch_2；Active-Contour-Loss.py 的 Active_Contour_Loss；Chan–Vese 的 _cv_calculate_averages（67–79 行）、_cv_difference_from_average_term（82–88 行）、图像缩放（332–335 行）。网页抓取未能展开 scikit-image 源码，已直接从上述官方 raw URL 成功读取，SHA 在收据。其[官方 API 文档](https://scikit-image.org/docs/0.25.x/api/skimage.segmentation.html#skimage.segmentation.chan_vese)和 [IPOL 算法及源码发布页](https://www.ipol.im/pub/art/2012/g-cv/)也支持该区别。

## 当前本地实现与原稿对应

[sammed3d_graymean_single972.py](../../../scripts/sammed3d_graymean_single972.py) 的 gray_loss 计算 sigmoid(logits) 对标准化 T1ce 的加权均值，分母为概率总和（下限 1e-6）；目标为真实整体肿瘤区域的灰度均值；逐例均方再求批均值。FirstPromptLoss 对首交互加权重 1 的灰度项，对全部 11 次交互加入各 0.5 的 Dice/交叉熵分割项。这符合一种合理公式实现，且灰度链路保持可微；但 soft mask、权重 1、归约、只加一次和强度尺度都是本地决定，不能称作者确认参数。

[预处理入口](../../../scripts/build_t1ce_wt_official15_cache.py) 使用 1.5mm、标签中心 128³、正值样本均值/标准差，归一化作用于整块。没有 N4 强度不均匀校正或直方图校准。原稿“intensity correction”未说明算法，故无法证明此点对齐。脑外背景经整体标准化可变成负数；灰度预测均值的软概率若在大量背景上不够接近 0，这些体素会进入分子与分母。是否实际导致损害需要训练样本诊断，不能仅凭代码定因。

## 可解释的风险与不能跳过的因果边界

1. 平均亮度相同不保证分割重合：两个位置完全不同但亮度相同的区域都可能令灰度项为零。这是目标函数本身的局限，不是实现专属 bug。
2. 对输入作 x'=a*x+b，区域均值差平方会按 a² 缩放。因而原稿未说明强度尺度和总权重，就不足以认为同一个“权重 1”代表相同约束力度；z-score 还会逐病例改变相对尺度。
3. soft 区域与最终二值区域不是同一个集合；软均值变好不保证最终二值 Dice 变好。分割损失和灰度损失的标量大小也不能代替参数梯度大小或方向。
4. 整体肿瘤由多个标签合并，区域均值不表达边界、空间位置或内部不同组织。输入模态、标签定义若不与论文一致，论文增益不可直接外推。本次未新增像素访问来验证这一机制。
5. 原稿表 2 的 baseline 描述不等于项目旧 0.594226 基线的真实训练身份。旧基线必须按本仓库检查点/日志核对；不能把它称为 Turbo 直接测试。旧新训练步数、选择指标等显著不同会阻止“只改灰度项导致掉点”的因果结论。

## 要消除作者复现不确定性，最少缺什么

需要师兄实际用的 loss 函数、调用位置、预处理配置、训练参数及论文表 2 对应任务/划分/初始化/检查点选择规则。可以按函数级最小材料索取，不需要对方交付整套私有病例。本轮未向任何外部联系人发送消息。

这份阅读没有提出或启动新的训练组，也没有选择新阈值。已停止的 161/200 轮状态保持不变。


## 用户指定 SAM-Med3D 官方库后的追加核对

本节围绕当前实际使用的 uni-medical/SAM-Med3D，不扩展到名称相近的新项目。核对本地已入库快照与当次官方公开代码；“不等于官方默认配置”和“论文未披露”分开记录。

### 真实灰度阈值 170 及调用链

[本地 click_method.py](../../../third_party/SAM-Med3D/utils/click_method.py) 第 7 行定义 get_next_click3D_torch_no_gt，函数内第 25 行把输入图像与 threshold（默认 170）比较作为可能的前景。它根据亮度候选和前次预测的差异生成正提示；没有候选时随机给负提示。这是没有真实标签时的经验选点启发，非区域灰度均值损失，也不参与网络反传。

在本地快照全部 Python 文件中搜索，该函数只有定义，没有实际调用者。train.py 第 23、63–64、221 行使用 get_next_click3D_torch_2，从真实标签误差区采点；inference.py 第 21、67–70、254 行及 validation.py 第 20、55–58、251 行使用真实标签驱动的距离/随机误差选点。当前 AFlow 交互函数也没有调用 170 灰度选点。不能因为这个未使用工具函数存在，就认定当前训练漏掉了官方默认步骤；也不能直接把 170 套到已 z-score 的 T1ce 图像上。

因此本库确实同时存在三种不同含义：170 为输入强度选点阈值；1000 为标签前景体素筛选；0.5 为预测 logit 的提示采点边界。师兄公式则是第四种内容：两个区域平均强度的误差。

### 官方训练默认值与本地灰度训练的确定差异

| 项目 | 已核官方代码/本地快照 | 灰度单组 runner | 判断 |
|---|---|---|---|
| batch 与累积 | train.py 51–52：每卡 batch 12，accumulation_steps 20；双卡名义有效 batch 480 | 后续授权双卡每卡 4、全局 8，无累积 | 明确不等于官方默认；用户授权大batch仅说明授权，不使其与旧有效batch相同 |
| 学习率 | train.py 135–143：图像编码器基准 8e-4，提示/掩膜解码器为 0.1 倍 | 8e-4 / 8e-5 | 分组数值对齐，但每次更新覆盖样本数不同，训练轨迹不能称对齐 |
| 分割损失 | train.py 126 使用 DiceCELoss 默认权重；本地安装 MONAI dice.py 674–675 的两权重默认均 1 | 两权重均 0.5，再加首提示灰度项 | 明确更改官方loss尺度；0.5来自MSCL公式4，灰度权重1属于未披露项的本地选择 |
| 验证与选best | train.py 358–359 的 eval_epoch 仅 return 0；train 中不调用它。保存依据是训练loss、训练Dice和滚动latest | 每5轮真实评估108验证，依最后第5点击Dice选best | 本地新增验证/选择机制，不能说“官方默认提供了同一验证最优模型” |
| 灰度均值loss | 官方训练函数没有它 | 首提示新增 | 本地研究改动 |

本地 train.py 的累积触发条件是 step%20==0 且 step!=0（step从0开始），不能只用名义有效batch估算旧运行更新数；确切步数应以历史实际日志和优化器记录审计。当前不建议为了“原样”追随上游的尾段处理，这会与每轮972例完整、不得丢弃或重复的用户约束冲突。

### 源码与预训练权重身份

[快照来源说明](../../../third_party/SAM-Med3D/SNAPSHOT_PROVENANCE.md) 已明确：复制日期2026-08-13，来源目录 SAM-Med3D-main 非Git仓库，精确上游commit未知。本地 third_party 下运行 git 会向上识别 AFlow.git；返回的 AFlow commit 3f457218fc716093fe53f6df8a5d5e6379d66346 不能冒充 SAM-Med3D 上游commit。本次重算本地 train.py SHA为474d9765e29c0196e9cf1da2242869c2cbd6970632d9f113357d38a8094e1e3f；click_method.py为0838972fb2f33de579a2652d3225f862d84dc6df113b152ea6018730c9b792d7。当前网上main已有变化，不以远端最新文件SHA替换历史绑定。

[官方README](https://github.com/uni-medical/SAM-Med3D) 说明 Turbo 是在44个数据集继续训练的版本，并链接[官方发布的Turbo权重页](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth)。本地 artifacts/checkpoints/sam_med3d_turbo.pth 本次重算 SHA256 为899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c，与该发布页显示SHA完全相同。这支持“用了真实官方Turbo初始化”，不证明额外灰度训练流程与上游默认相同。

以上均为只读代码、字节哈希与公开发布信息核对，没有模型前向或新队列访问。


### 不同源码 SHA 的进一步核查

为避免把格式差异误判成训练逻辑变更，对本地快照及本次读取官方 main 的 train.py、utils/click_method.py 进行了抽象语法树比较（忽略位置、空白、注释和函数说明字符串，不运行远端代码）。[比较收据](PUBLIC_CODE_AST_COMPARISON.json) 列出两个版本的SHA及所有函数匹配状态。

train.py 的所有函数中只有 init_checkpoint 存在逻辑表达差异：远端 torch.load 显式增加 weights_only=False；set_loss_fn、set_optimizer、interaction、train_epoch、eval_epoch、train 等函数的语法结构相同。点击代码所有函数在去掉说明字符串后相同，包括170强度选点、0.5输出阈值、距离选点及随机误差选点；整文件还有导入顺序和排版差异。两个文件SHA不同并非隐藏灰度算法变更证据。

该检查证明上述特定函数与本次官方main在语法层面对齐，不能补造未知上游commit，也不覆盖所有第三方文件或运行环境。Turbo权重身份相同、特定函数对齐，仍不代表本地新增灰度runner的配置、验证、batch和训练轨迹保持官方默认。
