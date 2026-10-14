# 空间往返诊断V2：固定原生体素坐标口径

2026-09-08，V2访问前登记。V1单位声明失败和部分fit访问已披露；仅fit684/313，无模型/开发/测试读数。参见AMENDMENT.md。

- 输入：fit SHA `75504df75fa734723945a9f07d76696fdcca89b8c29563dc0ad02a2560718d61`；cache index SHA `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`。原生label SHA逐例使用旧fit-only AUX_CACHE_RECEIPT，收据SHA `cb7a89d933e7fd3f21250ba002b9c082c80018274b800c5f820ea7eed8be9554`。完整labels.u8只作SHA与只读映射，语义取样限fit684。
- header：读取fit影像header而不物化影像dataobj；按orientation变换推导canonical shape/affine，与canonical label核对shape exact、affine atol1e−5/rtol0。要求RAS、有限非零spacing、正交轴误差≤1e−6，记录单位字符串及affine数值轴长度；unknown不猜毫米，所有距离统一原生体素坐标。
- 标签：raw整数且值域⊆{0,1,2,3,4}，WT取1/2/3。全原生mask和cache中的固定image-only crop，原生WT非空；非空和大小检查不用于剔除低保真病例。
- downsample：原生WT crop→float32→PyTorch nearest-exact→128³，结果须与对应旧cache uint8完全相同。用原生26邻接连通块ID相同nearest-exact取样；ID必须可用float32精确表示，ID>0应与二值cache完全一致。
- crop-only参照：将未缩放的WT crop放回全原生shape，报告原生WT数、crop内/外数及crop-only Dice。crop丢失非零仍继续完整诊断。
- restore N：128³二值图→nearest-exact至crop extent→放回完整原生shape。
- restore L：128³二值指示图float32→trilinear、align_corners=False至crop extent→严格>0.5→放回完整原生shape。不是模型输出、不叫模型oracle上限。
- 每种restore：原生全图Dice、crop内部Dice、交集/pred/target计数、FN/FP体素、带符号与绝对相对体积误差；surface-distance库按spacing=(1,1,1)计算原生体素坐标HD95（95%表面距离）、双向平均表面距离和一个原生体素容差的表面Dice。预测空时全图Dice0、表面Dice0、距离null，并单独计数；不以0替代无穷距离。
- 连通块：固定26邻接；原生大小分组1、2–8、9–27、28–125、126–1000、>1000体素。报告全部原生组件、完全/部分crop、在128中有/无自身ID的组件；crop内仍有体素但128无ID才计resampling loss。记录128二值连通块与原生ID的关联，区分多个原生块并入一个低分辨率块和一个原生块分裂；两种restore在原生块内的重叠只作覆盖描述，可能来自相邻块，不能代替ID存活判据。
- 汇总：原始顺序全684行；Scan均值、患者内平均后等权Subject均值、min/p05/median/p95/max；组件汇总同时报告分母与完全消失数。无bootstrap、无模型排名、无优化阈值。首例在worker1预检及正式worker进程重复，除时间/RSS外整个计数结果须一致。
- 冻结/输出：通过合成测试后写PREACCESS，绑定IDEA、PREREG、脚本及库实现/fit/index/源hash收据。先固定首fit例，再4 workers覆盖全684；RAW_CASES.jsonl、PROGRESS.json、RESULT.md/JSON、VERIFICATION.json及ARCHIVE_SEAL保存。源身份/重建失败写FAILURE，不覆盖或自动重启。全流程1800秒、raw报告64MiB上限；不创建重建图像缓存。
- 结果解释：往返保真度不是信息论上限，也不是模型Dice。不得由训练标签诊断改写旧测试排名或执行GT裁剪/病例排除。诊断完成后据证据决定下一工程步骤，弹性形变与其他候选另行正式预注册。

- V2新增：仅允许header单位mm或unknown，保持全病例统一体素距离；每worker日志记录header/label-array开始与完成。PREACCESS额外绑定V1 FAILURE、HEADER_UNITS_AUDIT及新单位适配测试。其余公式和停止规则不变。
