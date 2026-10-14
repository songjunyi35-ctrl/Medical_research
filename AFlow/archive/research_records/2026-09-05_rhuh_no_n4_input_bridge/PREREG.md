# RHUH no-N4输入桥接冻结协议

状态：2026-09-05 UTC，在任何RHUH标签下载、模型输出或Dice/Point之前冻结。

## 固定来源与角色

- 当前唯一角色是input-only feasibility，不是validation/test。固定TCIA package684官方后两个时点80 scans/40 subjects；病例不因影像或未来标签内容排除。
- `RHUH_IMAGE_QC_PREACCESS.json` SHA256=`65892e4d90a577ff2623cd2786d5142aeed86be809a5b273faf245321a4481c9`。
- `RHUH_IMAGE_DOWNLOAD.json` SHA256=`c6f17decbc81e8a2bc8ec385438b7e83547644aed26009fb376bd27a218d725f`，80/80官方MD5/bytes通过。
- 旧失败格式报告SHA256=`de0689c4008316b948f58bf4c07f439ca459deafc9d13468727c32ca7ca2849c`，只证明原非负门失败。
- 发布者官方预处理仓库固定commit=`46cf3825278243036c08131b62dcfcfb1d3bba39`：配准→SynthStrip→CaPTK z-score，无N4。

## 冻结变换与工程门

1. 每个下载条目先重算SHA256并与下载收据匹配；nibabel canonical orientation后必须为finite 3D、正spacing。
2. 背景解码固定复用`sammed3d/background_zero_decode.py`：六面众数比例≥0.95且等于global minimum，float64减offset后转float32，正支持必须exact等于原图`>background`。
3. 以decoded>0取image-only bbox；在整份decoded正支持上用float64计算population mean/std，并在float32图上标准化。Crop后trilinear resize到128³，decoded support最近邻resize并把mask外清0。不运行N4。
4. 每例要求背景exact0、support/bbox非空、source normalized mean绝对值≤1e-6、population std误差≤1e-6、输出shape=`[1,128,128,128]`且finite、输出brain mask非空。
5. 首3例按固定下载收据顺序再次独立计算，最终image和mask SHA256必须exact。
6. 80/80全门通过才允许写`status=passed_input_bridge`；任何失败整体关闭，不改门/算法/病例。

## 后续访问锁

- 本脚本不得打开、下载或stat RHUH segmentation路径，不加载任何checkpoint，不import模型，不计算Dice/Point。
- 即使格式通过，也不自动授权外部评测。只有no-N4 E/F两个fold、identity/Flip8全部预注册性能门通过，条件Full972 no-N4训练完成并冻结后，才可另写外部preaccess收据并下载全部80标签。
- 未来标签编号不用于挑病例。RHUH Whole Tumor冻结为所有positive published tumor labels的union；geometry/整数/允许域须全量通过后才推理。
- 未来primary固定Flip8、threshold0、Point=argmax、无SAM/后处理；Scan/Subject Dice与Point报告，患者簇bootstrap10,000次、seed20261361。绝对描述门预定Scan≥0.70、Subject≥0.70、Point≥72/80；无论正负都只评一次并归档。
- 不访问或重测fixed validation、outer、confirmation、test270、UCSD75；RHUH不回流选择模型/epoch/阈值/view/后处理。

## 资源停止门

- 当前CPU墙钟≤20分钟，新增持久文件≤20MB，GPU/模型forward/标签访问均为0。
- 若超限或已有输出路径存在，fail closed；不覆盖历史报告。
