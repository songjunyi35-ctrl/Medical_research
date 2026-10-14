## 自动工作流固定内部确认完成：双模型联合门通过=False

动机：将六折策略收束成两个固定部署workflow。仅按开发视图集合取中位代表：seed11[0,5]两视图，seed12[0,5,6]三视图；权重/输入/融合/阈值/点固定，无新训练。复用非盲test270/119六输出全部已入MD/JSON榜，不择test赢家改方案。

Seed 2026090711 pruned：Scan/Subject=0.744795478/0.733333687、Point257/270；对full8 Scan差-0.003270、Subject差-0.002708、Point-5。门{'fixed_forwards_at_most4': True, 'point_floor': False, 'scan_ci_floor': False, 'scan_loss_at_most003': False, 'subject_ci_floor': False, 'subject_loss_at_most003': True}。

Seed 2026090712 pruned：Scan/Subject=0.744617935/0.729756966、Point258/270；对full8 Scan差-0.002300、Subject差-0.002558、Point+0。门{'fixed_forwards_at_most4': True, 'point_floor': True, 'scan_ci_floor': True, 'scan_loss_at_most003': True, 'subject_ci_floor': True, 'subject_loss_at_most003': True}。

直觉判断依全部原门：开发通过未稳定转为固定内部确认，不能择seed或改门救结果；工程加速不代表精度/点安全。

本次5670前向、131.4秒含归档，0训练；/tmp新增logits12.656GiB+mask405MiB保留。540例在线parity和1620个保存输出独立核验；部署2/3×，既有fit阶段加速3.96/2.65倍，不含预处理。已新访问reused test，旧outer/confirmation/固定validation/外部未访问。本面板关闭，不测试回流调参。全榜总体/单次Scan最高0.757269662/0.752003176。

[七问](archive/research_records/2026-09-11_automatic_workflow_confirmation/IDEA.md)、[完整结果与CI](archive/research_records/2026-09-11_automatic_workflow_confirmation/RESULT.md)、[原始报告](archive/research_records/2026-09-11_automatic_workflow_confirmation/RAW_TEST270.json)、[独立核验](archive/research_records/2026-09-11_automatic_workflow_confirmation/VERIFICATION.json)、[榜单更新](archive/research_records/2026-09-11_automatic_workflow_confirmation/LEADERBOARD_UPDATE.json)。
