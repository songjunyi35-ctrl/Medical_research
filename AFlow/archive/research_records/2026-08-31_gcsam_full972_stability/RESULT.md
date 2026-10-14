# Full972 GCSAM跨seed稳定优化结果

## 结论

GCSAM（梯度中心化的锐度感知最小化）在fresh train-only inner上的平均Dice略高于完全配对的AdamW，但没有缩小病例/subject不确定性，两个冻结CI安全门都失败。按预注册停止，没有运行fresh-seed full972 final或fixed validation108。用户随后明确授权对冻结的fit864 train-inner-selected epoch184 checkpoint运行一次failed-gate exploratory reused test270。

Candidate只按post-100 Scan选择epoch184，Scan/Subject/Point为`0.753587/0.736140/102/108`；same-epoch AdamW为`0.752780/0.733582/102/108`。Candidate相对Control为Scan `+0.000807`、Subject `+0.002558`、Point持平，但paired 95% CI下界分别为`-0.005464/-0.005327`，均低于冻结`-0.003`门。

通俗地说：平均分略涨，但不同患者有人明显变好、也有人明显变差，结果没有变得更可靠。该方法要解决的正是跨seed/病例稳定性，因此不能只拿正的平均数放行。

## 链路与唯一变化

完整链路审视后，本实验只改变student参数更新：

- 两臂都使用同一Full972 train角色、fresh subject-disjoint `864 fit / 108 inner tune`、同一seed20261002、初始state、每轮sampling indices和首批增强tensor。
- 模型均为4,700,897参数plain SegResNet；teacher均为`0.75 history168 + 0.25 EMA168`；loss均为DiceCE加`0.3×T²`逐体素KL、`T=2`。
- Control使用历史AdamW一次student forward/backward。
- Candidate固定`rho=0.05`、non-adaptive：第一次反传后按论文公式中心化多维参数梯度并扰动student；同一batch、同一teacher logits、同一dropout RNG做第二次反传，精确恢复参数后由同一AdamW更新。
- 阈值0、无Segment Anything Model、无测试时增强、无后处理，部署仍为1×普通student。

## 实现与工程证据

- 新split：864 scans / 394 subjects fit；108 scans / 50 subjects inner；case/subject零重叠。
- CPU optimizer合同7项通过。
- CUDA BF16 `rho=0` parity：loss、clip前gradient norm、更新后参数精确0差。
- 双卡真实batch V2 smoke：初始state、sampler indices、增强后的image/label、first-pass loss exact。
- 正式双臂均200轮、每轮968 consumed draws / 121 steps；所有epoch sampler index SHA逐轮exact。
- 实际GCSAM扰动范数稳定为约`0.05`；无NaN、teacher gradient、restore/reload或访问越界。

## Train-only inner结果

| 方法 | selected epoch | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|---:|
| AdamW same-epoch control | 184 | `0.752780355` | `0.733581893` | `102/108` |
| **GCSAM candidate** | **184** | **`0.753587272`** | **`0.736140191`** | **`102/108`** |
| Candidate − Control | — | `+0.000806917` | `+0.002558299` | `0` |

配对分布：

- Scan 95% CI `[-0.005464039,+0.006866152]`；58/108改善、49变差、1持平。
- Subject 95% CI `[-0.005327105,+0.009841399]`；26/50改善、23变差、1持平。
- Point持平102/108。
- Bootstrap固定10,000次、seed20261004。

Control自己的post-100最高点是epoch183 `0.754562/0.734844`，但预注册要求由Candidate独立选epoch并只比较same-epoch control，因此不能用Control最佳epoch替换正式比较，也不能手选Candidate的其他Point更高epoch。

## Promotion与科学解释

通过：

- Candidate Scan/Subject点估计严格超过same-epoch Control。
- Point下降不超过1/108（实际持平）。
- 初始state、sampling weight、全部epoch index、首批增强tensor exact。
- 双臂全部工程、finite、memory、time、reload与access门。

失败：

- Scan CI下界`-0.005464<-0.003`。
- Subject CI下界`-0.005327<-0.003`。

因此`final_retraining_authorized=false`。

直觉只得到很弱的局部支持：GCSAM在同一inner seed的平均Scan/Subject都略高，说明paper-form更新确实能找到不同且平均略好的解；但病例和subject改善/变差接近对半，CI反而很宽，没有证据表明它解决了稳定性瓶颈。这个结果只否定当前`rho=0.05`、non-adaptive、Full972 T1ce/WT、200轮实现；不证明所有sharpness-aware optimizer不可能有效。但按冻结规则不得搜索rho、centralization维度、adaptive flag、clip、epoch或seed。

## 用户授权failed-gate reused test270

用户在得知inner CI门失败、fresh-seed final未运行后明确说“可以去test上评测了”。测试前冻结了GCSAM checkpoint、inner decision、实现收据、test manifest/cache、当前内部最高Full972 KD168报告、阈值0、1×forward、无TTA/SAM/后处理和一次性evaluator哈希。

严格same-epoch AdamW checkpoint没有保留（runner只保存各arm窗口最佳；AdamW保存epoch183，GCSAM选中epoch184），因此没有把epoch183冒充same-epoch control，也没有额外重训。测试只评GCSAM，并与archived Full972 KD168配对比较。

| Reused internal test270 / 119 subjects | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| **GCSAM fit864 inner-selected epoch184** | **`0.752003176`** | `0.737266420` | `252/270` |
| Full972 random-init voxel-KD epoch168 | `0.751759537` | **`0.737327107`** | **`262/270`** |
| GCSAM − Full972 KD | `+0.000243638` | `-0.000060687` | `-10/270` |

- GCSAM自身Scan/Subject bootstrap 95% CI为`[0.733930,0.769141]` / `[0.710739,0.761634]`。
- 相对Full972 KD的paired Scan 95% CI为`[-0.005655,+0.005740]`。
- 相对Full972 KD的paired Subject 95% CI为`[-0.009153,+0.008094]`。
- 150/270 scans改善、120变差；59/119 subjects改善、60变差。

按Leaderboard的主排序Scan Dice，GCSAM是新的reused test270自动direct locator Scan点估计第1，比Full972 KD高`0.000244`；但差异CI跨0，Subject略低且Point少10，不能称三指标全面最佳、显著提升、Full972 final、盲测或外部确认。测试结果禁止触发rho、epoch、seed、阈值或postprocess调整。

## 成本

- AdamW：`14,997.90s`（4.166h），peak allocated `20.292GiB`。
- GCSAM：`23,306.19s`（6.474h），为Control训练时间`1.554×`，peak allocated `20.358GiB`。
- 两卡并行总墙钟约6.47h。
- Workspace只有split、报告和少量checkpoint，不建新cache；推理成本没有变化。

## Evidence与访问

- Inner decision SHA-256 `78851d7b15bc4414e404f968dce816241c3d7656fc27c80250059e1a86341577`。
- Control/Candidate TRAIN_DONE SHA-256 `8b080bd8c4b24c23be35f4085c17e984b3ed9736252c6f43e846c1e329f5fd58` / `9c53d2ef4a7457e85a59b36e59227cb9fdf2265d835c9e66f9b81d768b32da5c`。
- Control/Candidate post-best checkpoint SHA-256 `5326326fa9caf249f16d7b378bb1ada047b43412f9488b55d3a9d41a30ac14c0` / `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`。
- Implementation receipt SHA-256 `74536ec022d2094732be68841b454ef0a32f57434c0c743fa9e221220cf175fb`。
- Test raw report SHA-256 `d0e2f48e1f499d65adddb3dbe30758caad27768bb75f737ea2cfe4a877c335d5`。
- User override receipt SHA-256 `527ee72332ce1bbc9edc18dd453230c9654eaa653de37fae91058589e31741df`。
- Train-only inner和reused test270已访问；fixed validation108未访问；outer未访问；confirmation未访问。

## 下一决定

停止当前GCSAM分支，不补跑fresh-seed full972 final，不调rho/epoch/seed/clip/阈值。Test只增加了一个`+0.000244`且CI跨0的Scan点估计，同时Point明显下降；该结果进一步表明“平均Dice略升”不等于“稳定性提高”。下一步必须取得独立新cohort或提出新的单因素机制，不能用本test继续优化。
