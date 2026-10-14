# MOTUM-HGG29冻结checkpoint面板：预注册

状态：冻结于cache、checkpoint加载和性能访问之前；29例源数组只用于来源/标签/geometry门。

## 冻结身份与来源

- IDEA SHA-256 `6327548760ddb153ec408a1562a1a10498ca724a4f55c2056dc2a611ffcb35f5`。
- Source audit SHA-256 `69d59768cd54ed526aa1cc8b0f66e31884ea4aaad171a1a16bc17c4488412932`。
- HGG29 cohort SHA-256 `f49e8f888b0fce8b7f34da1f9eadb49039d10aaed8e0538008f215df1b953379`。
- 官方DOI快照`10.12751/g-node.tvzqc5`：984,434,960 bytes；发布者无checksum，两次独立完整传输SHA-256均`2626862599a3fcfe4ac0cefcea3af5b190625275036cc8eb4c9039cbd54e2d7c`、byte-exact、双CRC通过。
- Participants选择固定：WHO CNS Grade∈{3,4}且Origin=`/`，sub-0001至0029；12 grade3/17 grade4；38名脑转移瘤永久排除。
- 证据：CC BY 4.0 public external-to-project；三家中国医院institution-disjoint/no-known-overlap；无patient-ID bridge；非hidden blind。

## 冻结数据与预处理

- 每例仅`sub-XXXX_t1ce.nii.gz`、`flair_seg_label1.nii.gz`、`t1ce_seg_label2.nii.gz`；三文件SHA逐例来自冻结cohort。
- 三者closest canonical RAS后shape exact、affine max error≤`1e-5`；image finite且有正支持；两个mask有限/近整数，FLAIR允许`{0,1}`、T1ce允许`{0,1,2}`。
- WT=`flair_mask>0 OR t1ce_mask>0`。一例T1ce mask为0/1而非0/2仍按语义文件正类处理，不排例。
- Canonical N4 shrink2、iterations `[50,50,50,50]`、SimpleITK threads1；只按N4后正值图像求bbox；source-positive z-score；128³ image trilinear/label nearest。
- 首例sub-0001独立probe后才允许全29 cache；全量结束须首例image/label byte exact、29 finite/binary/nonempty、source/hash/label union重验。

## 冻结模型与统计

- Full972 KD168 SHA `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`：identity+固定8 reflection ordered float32 raw-logit mean。
- GCSAM184 SHA `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`：identity。
- Views顺序：`(),(-1,),(-2,),(-2,-1),(-3,),(-3,-1),(-3,-2),(-3,-2,-1)`；batch8、BF16 autocast、threshold`>0`、Point=argmax、无SAM/后处理/模型ensemble。
- 29×9=`261` forwards；Full先、GCSAM后。
- 绝对3行：Scan/Subject Dice、Point；每人一scan所以Scan=Subject。
- 配对2项：Flip8−Full identity、GCSAM−Full identity；报告improve/worse/tie和Point差。
- Bootstrap 10,000，base seed `20261481`：Full/Flip8/GCSAM absolute偏移0/10/20，Flip8/GCSAM paired偏移100/200。
- 解释：delta>0且Scan CI lower>0=`supported_on_MOTUM`；delta>0且CI跨0=`positive_but_uncertain`；delta≤0=`not_supported_or_reversed`。不设promotion。

## 停止、访问与代价

- Source/cache/smoke/checkpoint/preaccess任一门失败即停止；禁止改病例、mask合并、N4、crop、resolution、view、阈值、bootstrap或容差。
- 正式started receipt一旦写入，结果无论正负都不重跑；不据MOTUM调参或访问旧cohort。
- 当前已访问29例source arrays；cache=false、checkpoint/model/performance=false。正式不访问internal/fixed validation/outer/confirmation/UCSD/Africa/CPTAC/WAW。
- 训练0；cache约304,087,040 bytes；261 forward；极厚层5.5–6.75mm/21–24 slices、半自动后医师校正和训练源parity caveat必须保留。
