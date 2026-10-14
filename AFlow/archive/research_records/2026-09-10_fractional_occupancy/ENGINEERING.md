# 占比target工程过程

04:57：11项合成测试通过，独立三轴格重叠矩阵与积分实现对齐；非整数下采样/上采样/常数/identity、float16质量误差及非法输入均覆盖。没有改原损失/阈值来套SoftSeg收益。

04:59–05:00：CACHE_PREACCESS写在原生fit标签读取前，继承684个source label SHA。缓存会话12958启动，仅CPU，不访问dev/test，不改原cache。

05:05：已96/684例、235.49秒，较事前“数分钟”估计慢，按当前吞吐约28分钟；包含原生标签读取、校验、解码和占比计算。没有为提速改hash-bound builder/算子，仍可在主弱形变训练结束前完成。最终成本以CACHE_RECEIPT为准。

训练入口已独立准备：只覆盖dataset第一标签通道，原图像/RC辅助零权重/抽样/初始化/原增强/168轮保留，未正式准入或启动。GPU batch8 smoke必须等主路线关闭且有可用GPU后再做，不能在当前40GB卡上硬塞另一个20GiB训练进程。

05:39完成builder（2356.10秒），05:44完成独立全文件/逐例重读核验。原生label SHA与旧硬cache684/684 exact；q64最大质量误差3.36e−16，q16最大4.02e−5/平均1.93e−6；硬化无空WT，与旧hard重叠均值0.973320。仅诊断指标，候选尚未训练/评dev。实际CPU成本39.27分钟，高于事前估计，RSS1.094GiB。Target SHA32d7815fbfc79e5bfd42e9872808b6afb439e324151068a812c79a0b49d51ee6，index SHA1d89c1c7d02eca3a458a61223d1f9eed40bb8bc8022a99b9c04fa5405413f182。15项算子/原损失兼容性/评估合同测试通过。

05:53–05:54数据入口CPU核查通过：固定fit前8例图像张量/RC通道exact不变，WT为float32训练张量且与存储q16总量exact；确认target实际改变、未加载模型/进行前向。DATASET_CPU_RECEIPT.json绑定入口和依赖SHA，后续训练准入须检查。该检查读取fit8缓存影像/标签，不是cache构造阶段的“0image pixel”访问范围，也没有dev/test访问。

08:12–08:15：弱形变主研究负结果正式归档后，按事前条件运行batch8 GPU smoke。初值/图像/RNG匹配、辅助梯度0、target确实改变更新、导出0、显存/时间门通过；TRAIN_PREACCESS/IMPLEMENTATION_FREEZE写在正式训练前，双seed会话39049启动，PID19199/19200。08:24前10轮独立抽样/原RNG/第一batch图像字节重放exact，见TRAINING_REPLAY_082351.json；开发读出/归档代码随后冻结为DEVELOPMENT_IMPLEMENTATION_FREEZE.json，没有读新candidate dev。

条件性固定seed11 test代码已准备，bootstrap采用事前TEST_PLAN固定2026091032患者簇10k；加入270MiB二值mask保存、0新增forward。原硬标签、阈值0、原输出/八镜像4输出及4320forward不变。已知历史test/流水线test-informed边界明确，不称本研究独立盲测；actual test仍未准入或执行。
