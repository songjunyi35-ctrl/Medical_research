# AFlow BraTS 项目说明

最后更新：2026-08-18 UTC

## 1. 科研目标

建立一条可严谨答辩的术后胶质瘤自动分割硕士论文科研路线，不宣称公开排行榜 SOTA。
当前任务是 BraTS 2024 post-treatment GLI，输入为单通道 T1ce，二分类目标为
`WT = {1,2,3}`，排除原始标签 `4`。

当前全自动主系统是一个小型 direct dense locator。GT 中心裁剪、GT 派生 prompt 和
交互式 SAM 结果仍可作为机制证据或 oracle 上界证据，但不能当作自动端到端结果。

## 2. 仓库与数据事实

- 仓库根目录：
  `/mlx_devbox/users/songjunyi.666/playground/AFlow`
- 已核验的原始 T1ce/seg 挂载位置：
  `/mnt/hdfs/sjy/datasets/brats2024_gli_t1ce_seg_v1`
- 原始集合：1,350 个 scan，来自 613 个纵向 subject；包含 1,350 个 `*-t1c.nii` 和
  1,350 个 `*-seg.nii`。
- 权威 972/108/270 manifest 根目录：
  `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/`
- 基础 train、validation 和 internal test 之间的 case overlap 与 subject overlap 均为 0。
- 自动 locator 开发协议进一步把历史 972 个训练/验证侧病例分成 756 train、
  108 selector 和 108 confirmation。除非获得新的外部 cohort，否则新的 inner 开发划分
  只能使用已授权的 756-case train role。

270 例 internal test 和 108 例 confirmation 都已经揭示。它们只能作为固定报告集，
不能作为开发集。若要提出真正盲测的泛化结论，现在需要新的外部 cohort。

## 3. 当前 benchmark 总账

下列结果采用不同协议；协议列是结果本身的一部分，任何报告都不能省略它。

| 结果 | Split 与指标 | 协议边界 | Dice |
|---|---|---|---:|
| upstream-literal vanilla SAM-Med3D | 270 internal test，scan best-of-5 | 官方 1.5 mm；GT 中心交互点击 | `0.594226` |
| 历史 brain-crop SAM C3 | 270 internal test，scan | image-only crop，但点击由 GT 派生 | `0.625359` |
| residual refiner epoch58 | 270 internal test，scan | GT 中心 oracle；N4；旧三点击 coarse 输入 | `0.703798` |
| robust residual ensemble | 108 validation，scan | GT 中心 oracle；在 validation 上选择；未测 test | `0.700419` |
| direct plain locator epoch99 | 270 internal test，scan | 全自动；固定 logit `>0` | `0.725820` |
| **direct plain locator epoch168** | **270 internal test，scan** | **全自动；固定 logit `>0`；无 SAM/TTA/后处理** | **`0.741204`** |

当前最好的全自动 checkpoint 是 epoch168：

`workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth`

SHA-256：
`f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`

它在固定 270-case 集上的 scan Dice 为 `0.741203949`，subject-balanced Dice 为
`0.723179486`，automatic point hit 为 `0.929630`。相对冻结的 epoch99 checkpoint，
scan 和 subject Dice 的配对变化分别为 `+0.015383882` 与 `+0.013766983`，两者配对
95% 置信区间的下界均大于 0。由于这个 internal test 已被更早的实验分支揭示，因此它是
项目内部纪录，不是纯净盲测或公开排行榜成绩。

## 4. 已有证据支持的科研结论

1. **Direct 自动 dense localization 是当前成功主线。** 在历史 development 集上，
   locator direct mask 达到 `0.716125`；将同一表征送入旧冻结 SAM point+dense 路径后
   反而降到 `0.599324`。直接输出 head 避开了已测得的 prompt/decoder 分布失配。
2. **Image-only brain crop 有效。** 在匹配的历史协议下，brain crop 将 scan Dice 从
   `0.571109` 提高到 `0.625359`，提升 `+0.054250`。
3. **延长训练并增加 late-cosine 收尾是一个全自动正结果。** 在新的 subject-clean inner
   轨迹上，前 100 轮最佳 epoch95 的 scan/subject 为 `0.723604/0.719629`，100 轮后最佳
   epoch168 为 `0.746335/0.744497`。固定 epoch168 的重训也改善了 outer selector 和最终
   固定 internal-test 读数。该设计证明的是“额外 epoch + late-cosine”的组合，不能把
   收益拆成其中某个单独因素。
4. **Residual refinement 提供 oracle 机制证据。** Robust-shift correction 将 GT 中心
   development 结果提高到约 `0.7004`，但它不是全自动结果。
5. **推理 workflow 搜索提供效率贡献，而不是 Dice 贡献。** Nested MCTS 相对
   24-forward reference 的 subject Dice 变化为 `-0.000336`，平均 forwards 从 24 降到 7。
   冻结 8-forward workflow 实测 refiner 阶段约 `2.78x` 加速，固定 1-forward 实测约
   `13.66x` 加速。Random 和 greedy control 也找到相近 Pareto 点，因此不能声称 MCTS
   优于简单搜索器。

## 5. 已关闭或已判负的分支

- 灰度/intensity residual 建模未通过匹配的预注册门槛。
- CAPG 的 centre head 没有改善 frozen-SAM endpoint；direct plain locator 更好。
- Error-gated refiner、Domain-SAGE conditioning、adaptive routing 和 boundary auxiliary
  都没有建立其所声称的机制收益。
- 把旧冻结 SAM decoder 重新接到强 locator 后，Dice 明显下降。
- 若没有新假设、新的干净划分和预注册匹配对照，不得在同一已揭示开发病例上恢复这些分支。

## 6. 面向硕士论文的证据主线

当前可以辩护的贡献候选为：

1. 一个采用 brain-crop 表征、固定全自动推理的 image-only、4.70M 参数 T1ce/WT direct
   dense locator。
2. 一套 subject-clean 训练与评估方法，证明 late-cosine 长时程带来可复现收益，并包含
   checkpoint 冻结、重载一致性和配对不确定性报告。
3. 一个强类型、可审计的推理 workflow 搜索与剪枝框架，展示明显的精度—成本 Pareto
   收益，同时如实说明收益来自 workflow 稀疏化，而不是已证明 MCTS 优于其他搜索器。

这些仍是项目层面的贡献候选，不是已经定稿的论文章节结论。仍需外部验证和干净消融来
增强证据。

## 7. 权威证据索引

- 精简结果总账：`AFLOW_BRATS_RESULTS.md`
- 当前进展周报：`AFLOW_BRATS_WEEKLY_REPORT.md`
- 权威详细交接：`AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md`
- Epoch168 完整记录：
  `archive/research_records/2026-08-18_plain_long_schedule/RESULT.md`
- Epoch168 最终 JSON：
  `workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/FINAL_TEST270_EPOCH168.json`
- MCTS 与剪枝记录：
  `archive/research_documents/2026-08-17_root_cleanup/MCTS_EFFICIENCY_CONTRIBUTION_RESULT_20260813.md`
  和
  `archive/research_documents/2026-08-17_root_cleanup/INFERENCE_PRUNING_RESULT_20260814.md`

引用更多小数位或提出因果结论前，必须核对底层 JSON 和 checkpoint 哈希。
