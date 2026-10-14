# 工程记录（非开发性能）

03:44 UTC：5项形变合成测试通过。fit固定前8例、base/elastic各6步GPU smoke全门通过，原初始化与旧seed11control exact，原增强与模型CUDA RNG逐步equal，导出error0，峰值22,187,487,744 bytes=20.664GiB。48样本9次形变，最大位移2输入体素，最大梯度行和0.2500000596（容差1e−6）。原GPU40GB，未复用旧80GB显存假定。无dev/保护集像素访问。

03:46 UTC：训练前收据与实现/数据/历史控制完整哈希绑定。监督器会话12554、首波seed11两进程3773/3774（仅描述启动时PID，后续以TRAINING_PROGRESS为准）。每轮前两轮约46s，初始四臂估算约4.3h墙钟，不据loss或dev改轮数。

9项形变/评估门单测通过。初次全repo测试842 passed、1 failed、91 warnings、53.92s，唯一失败是旧/tmp/aflow-remind-dicomseg-deps固定wheel随环境丢失，不是本方法代码失败。按旧SHA从官方PyPI恢复pydicom==3.0.1 wheel并机械解压；不pip install、不改venv、不改旧测试或hash-bound脚本。wheel SHA=db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41，解压manifest SHA=49e2b56e2fac41da819f22e8801eb5b7afa5437f97f5eb48ddf3af43574c9946，均exact，未访问旧ReMIND患者影像。后续全repo复测另记录。

条件性full864清单只做metadata union：864/391人，source fit/dev固定SHA，清单SHA=d92619562fb301509f9668873684a3a7f57b0edee1df1fc35e65cea43bdbba52；未因此准入训练/改变现行Stage A划分。

04:00 UTC复测：全repo846 passed/91 warnings/48.88s。新增full864合成检查证明辅助系数0时真实RC与零占位的每个参数梯度bit-exact，manifest union864/391与患者权重归一检查通过。full864尚未GPU smoke/正式准入。

04:36 UTC复测：全repo857 passed/91 warnings/48.52s。04:37在形变模型尚未dev访问时，额外冻结评测/独立归档实现DEVELOPMENT_IMPLEMENTATION_FREEZE.json；新增DEV_ACTIVE_BATCH/DEV_PROGRESS故障边界记录，不改参数、评价或选择门，9项对应单测再次通过。

恢复边界：原training engine的recovery_latest.pth含原torch/CUDA/sampler/optimizer状态，但不直接序列化新增形变generator；后者可由固定seed+100000及已执行step确定重放。当前入口没有自动resume机制、不会因为recovery存在而悄悄重启同路径；若出现中断须先记录partial/access并另核工程恢复方案。当前监督器仍运行，不重跑入口。

硬件/对照边界：当前A10040GB、torch2.7.1+cu126/MONAI1.5.2/NumPy2.4.6/SciPy1.17.1/nibabel5.4.2/SimpleITK2.5.5已记录ENVIRONMENT_20260910.json。历史80GB control本轮复用而未从头重训；当前初值/原RNG/抽样匹配、两旧native checkpoint在dev180逐例计数和Point重现exact，不能扩张为整条历史训练轨迹已在新硬件逐比特复现。短时独立cleanup推理曾共用GPU0约32秒，训练未停、参数未改；运行成本不把硬件/共用影响全部归为形变算子开销。

05:57第一seed两层完成：native7767.617s、subject7763.088s、各20.664GiB、导出error0。各162624输入/20328steps、32531样本形变20.0038%，位移≤2、梯度最大0.2500002384，incoming_empty_selected=0。checkpoint SHA native12bd86129ac5d387c4ffd608012f7aabdccbb30e2b5780b90c24e8f1455683d7、subject273537ab9cc0de7e2bcd97ce1885ec9319436275f4f30ebc897c17ac41775e65。TRAINING_REPLAY_055913.json独立重放第一seed全部168轮和第二seed前2轮，全部exact。第二seed两层PID11562/11563继续，尚无形变新dev访问。

06:06全repo复测872 passed/91 warnings/47.46s；06:28两seed12训练均40/168，first seed全部完成。标签占比cache/数据入口全部CPU准备完成，但没有并入当前训练，也未启动该备选新模型。主实验评估实现已事前冻结，继续等待四臂齐全。

06:44–07:09条件性test输出工程：新增C序/little-bitorder二值mask存档（4×270×128³/8=270MiB，不增加forward），后续可从实际mask重新核交集而非只看计数字段自洽。4项存档测试、2项mask/GT核验测试、3项榜单渲染保留/排序测试通过；通用核验/归档脚本已准备，未执行真实test、未改实际榜单。只改未来审计输出、不改模型或成功门；现有弱形变dev评测冻结保持原字节。mask不能独立恢复logit argmax，该边界明确保留。

07:13复测全repo881 passed/91 warnings/48.50s；07:28工作满4小时，seed12两层118/168，仍无本方法新dev访问。监督器12554唯一活跃科研作业，计划11:28:19 UTC收尾时间不变。
