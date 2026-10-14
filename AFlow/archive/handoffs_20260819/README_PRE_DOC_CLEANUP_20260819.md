# AFlow × SAM-Med3D：BraTS 2024

本仓库目前只有一条生效的 BraTS 科研主线：**T1ce -> WT={1,2,3}**，排除原始
标签 4。开发协议采用 subject-clean 的 972/108 划分和 `128 x 128 x 128`
输入。已经建立的 vanilla 基线使用严格 upstream 11-pass 训练循环和五次循环
mask 评测点击。

运行实验前必须先读
[`AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md`](AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md)，
它是唯一权威交接文档。精简数字见 [`AFLOW_BRATS_RESULTS.md`](AFLOW_BRATS_RESULTS.md)，当前
进展见 [`AFLOW_BRATS_WEEKLY_REPORT.md`](AFLOW_BRATS_WEEKLY_REPORT.md)。已过时
叙述保存在 [`archive/handoffs_pre_20260813/`](archive/handoffs_pre_20260813/README.md)。

## 当前命令

worker wrapper 只暴露 T1ce 模式：

```bash
bash scripts/run_worker_research.sh probe
bash scripts/run_worker_research.sh doctor
bash scripts/run_worker_research.sh sammed3d-t1ce-wt ...
bash scripts/run_worker_research.sh eval-sammed3d-t1ce-wt ...
bash scripts/run_worker_research.sh reproduce-official15-t1ce-wt ...
bash scripts/run_worker_research.sh cache-refiner-t1ce-wt ...
bash scripts/run_worker_research.sh rebuild-refiner-cache verify
bash scripts/run_worker_research.sh train-refiner-t1ce-wt ...
bash scripts/run_worker_research.sh cache-mcts-atoms-t1ce-wt ...
bash scripts/run_worker_research.sh promote-mcts-atoms-t1ce-wt
bash scripts/run_worker_research.sh search-mcts-workflow-t1ce-wt ...
```

使用未知 mode 调用 wrapper，可以查看 DPD-SAM T1ce 重建和 partial-unfreeze
等其他 T1ce 模式。

## 当前目录结构

| 路径 | 用途 |
|---|---|
| `sammed3d/t1ce_*.py` | T1ce 数据、prompt、cache 和训练约定 |
| `sammed3d/dpdsam_binary_model.py` | T1ce DPD-SAM 重建组件 |
| `scripts/*t1ce*.py` | T1ce 构建、训练、评测和审计入口 |
| `workspace/BraTS/sammed3d_t1ce_wt/` | 主线 checkpoint、日志、预注册和负结果 |
| `workspace/BraTS/cache/t1ce_wt_*` | T1ce 开发集 memmap cache |
| `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/` | 已核验的 HDFS 版 972/108/270 manifest |

270 例集合已经揭示，不是开发集。禁止用它选择 checkpoint、loss、阈值、crop 或
后处理。若要声称“未触碰测试集”，需要一个全新的外部集合。

## MCTS 推理 workflow 搜索

固定三个无灰度 loss 的 robust refiner checkpoint 后，可用真正的 UCT MCTS 搜索
推理 workflow。实现、证据边界和 GPU/CPU 两阶段命令见
[`archive/research_documents/2026-08-17_root_cleanup/MCTS_T1CE_WT_WORKFLOW.md`](archive/research_documents/2026-08-17_root_cleanup/MCTS_T1CE_WT_WORKFLOW.md)。搜索只接受 108 例
validation atom cache；CLI 不提供 test 模式，并按 subject 做 nested search / selector /
outer-holdout。结果仍是 GT-centered oracle validation 开发分析，不是自动定位或 test。
2026-08-13 的首轮三折结果见
[`archive/research_documents/2026-08-17_root_cleanup/MCTS_T1CE_WT_RESULT_20260813.md`](archive/research_documents/2026-08-17_root_cleanup/MCTS_T1CE_WT_RESULT_20260813.md)：没有得到 OOF
Dice 提升，但发现了显著降低推理 forwards 的精度—成本折中。

2026-08-14 已把该方向补齐为可作为硕士论文贡献报告的、精度约束且可审计的推理
workflow 自动优化框架。三折 MCTS 在 subject Dice 上相对 24-forward locked reference
变化 `-0.000336`，95% CI 跨 0，同时把平均 member-view forwards 从 24 降到 7；
但等预算对照中 random 和 greedy 也取得了相近 Pareto 点，因此不能声称 MCTS 优于
简单搜索器。冻结的 8-forward 工程候选经过真实在线交叉 GPU 基准，refiner 阶段
延迟由 `1.0757 s/case` 降到 `0.3871 s/case`，约 `2.78x` 加速。完整结论、限制和
证据哈希见
[`archive/research_documents/2026-08-17_root_cleanup/MCTS_EFFICIENCY_CONTRIBUTION_RESULT_20260813.md`](archive/research_documents/2026-08-17_root_cleanup/MCTS_EFFICIENCY_CONTRIBUTION_RESULT_20260813.md)。

随后完成的 1→8→24 adaptive router 没有通过等成本随机路由控制，因此已记为负结果；
正收益来自固定 workflow 剪枝。1-forward 在 validation 上相对 locked subject Dice 变化
`-0.000358`，双 A100 实测获得 `13.66x` refiner-stage 加速；4/8-forward 的 Dice
点估计略高于 24-forward locked，但配对 CI 跨 0。最新固定预算曲线、adaptive 负结果、
在线 benchmark 和论文表述见
[`archive/research_documents/2026-08-17_root_cleanup/INFERENCE_PRUNING_RESULT_20260814.md`](archive/research_documents/2026-08-17_root_cleanup/INFERENCE_PRUNING_RESULT_20260814.md)。

## 2026-08-14 架构实验

显式 error-supervised correction gate 的两 seed 实验没有超过 robust refiner，已作为
负结果记录。随后冻结 `0.699043` robust base，并训练零初始化的 uncertainty-conditioned
safe adapter：plain/gated validation scan Dice 分别为 `0.701991/0.702090`，subject
Dice 为 `0.697102/0.697140`。adapter 有正收益，但 gate 相对 plain 的额外差仅
`+0.000099`，不能宣称 gate 有效。完整证据和运行中断边界见
[`archive/research_documents/2026-08-17_root_cleanup/SAFE_ADAPTER_PAIR_RESULT_20260814.md`](archive/research_documents/2026-08-17_root_cleanup/SAFE_ADAPTER_PAIR_RESULT_20260814.md)，前一轮
负结果见 [`archive/research_documents/2026-08-17_root_cleanup/ELPGR3D_A2_RESULT_20260814.md`](archive/research_documents/2026-08-17_root_cleanup/ELPGR3D_A2_RESULT_20260814.md)。这些仍是
GT-centered oracle development validation，不是 official15 自动流程或 test。

safe adapter 随后在 seed17 full 972/108 上复现为 scan `0.702191`；seed16/17 plain mean
为 `0.702091`。同 seed 的 Dice-only boundary auxiliary 得到 `0.702632`，但只比 matched
control 高 `+0.000440`，未达到预注册门槛且 subject CI 跨 0，因此不作为额外正贡献。
完整结果见
[`archive/research_documents/2026-08-17_root_cleanup/BOUNDARY_SAFE_ADAPTER_RESULT_20260814.md`](archive/research_documents/2026-08-17_root_cleanup/BOUNDARY_SAFE_ADAPTER_RESULT_20260814.md)。

随后参考DPD-SAM的latent-domain conditioning实现Domain-SAGE 3D。fit-only域分类可以
学到validation accuracy `0.9537`，但Domain arm scan Dice `0.701765`低于同seed plain
control `0.701897`，Domain−plain subject差`-0.000245`且CI跨0，按预注册判为负结果。
plain seed18使SAGE完成第三个训练seed复现，三seed scan mean为`0.702027`。完整结果、
200-epoch解释和证据哈希见
[`archive/research_documents/2026-08-17_root_cleanup/DOMAIN_SAGE_RESULT_20260814.md`](archive/research_documents/2026-08-17_root_cleanup/DOMAIN_SAGE_RESULT_20260814.md)。

## Automatic locator / CAPG-3D

The first genuinely image-only brain-crop localization path is documented in
[`archive/research_documents/2026-08-17_root_cleanup/CAPG3D_RESULT_20260814.md`](archive/research_documents/2026-08-17_root_cleanup/CAPG3D_RESULT_20260814.md). A 4.70M-parameter
plain locator reaches `0.716125/0.706215` validation scan/subject Dice without
GT crop placement or GT prompts. CAPG's centre head reaches locator scan
`0.719090`, but does not improve the preregistered frozen-SAM point+dense
endpoint (`0.597910` vs plain `0.599324`) and has a much lower point-hit rate.
Treat CAPG as a negative ablation and the direct plain locator as a development
candidate; the 270-case test was not accessed.

## 历史材料

四模态、T2-FLAIR/WT1234、多目标、Paper1913、MedTAWS、GRPO、旧 prompt
baseline 和已过时的交接叙述都不是当前科研路径。恢复任何旧分支前，必须先核对
当前交接文档和 artifact 哈希。

2026-08-13 的文档清理记录在
[`HANDOFF_CLEANUP_20260813.json`](HANDOFF_CLEANUP_20260813.json)。旧交接文档
以及已经过时的 2026-08-10 清理快照保存在
[`archive/handoffs_pre_20260813/`](archive/handoffs_pre_20260813/README.md)。
当前代码依赖保护区、已清理内容和测试结束后的第二阶段归档计划见
[`archive/research_documents/2026-08-17_root_cleanup/CODE_CLEANUP_AUDIT_20260813.md`](archive/research_documents/2026-08-17_root_cleanup/CODE_CLEANUP_AUDIT_20260813.md)。

## 2026-08-18：plain locator 长时程

新的subject-clean inner实验表明100 epoch会截断本轮plain locator轨迹。预注册的
100轮固定`2e-4`加late cosine在inner上把scan/subject从`0.723604/0.719629`提高到
`0.746335/0.744497`。锁定epoch168后，在原756 train cases上从头训练且不读取
validation，单次outer selector达到`0.731008/0.714314`，相对epoch99 baseline提高
`+0.010589/+0.009697`，point hit变化`-0.009259`，全部outer门槛通过。

新的development checkpoint为
`workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/plain_fixed_epoch168.pth`
（SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`）。
confirmation和270 test未访问；旧epoch99仍是唯一有270 test结果的checkpoint。完整结果、
限制和证据哈希见
[`archive/research_records/2026-08-18_plain_long_schedule/RESULT.md`](archive/research_records/2026-08-18_plain_long_schedule/RESULT.md)。

## 2026-08-18：epoch168 固定 test270

随后用户明确授权对已固定epoch168做一次最终270测评。scan/subject/point hit达到
**`0.741204/0.723179/0.929630`**，相对epoch99提高
**`+0.015384/+0.013767/+0.029630`**；scan/subject配对95% CI分别为
`[+0.009186,+0.021574]`和`[+0.004123,+0.022935]`。

这是当前复用internal test上的历史最高点估计；test未参与模型、epoch或阈值选择，结果
不回流调参。由于270此前已经揭示，仍不是pristine blind test或外部确认。完整结果已合并
到[`archive/research_records/2026-08-18_plain_long_schedule/RESULT.md`](archive/research_records/2026-08-18_plain_long_schedule/RESULT.md)，
最终JSON为`workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/FINAL_TEST270_EPOCH168.json`。
全仓回归为`273 passed, 24 warnings`。
