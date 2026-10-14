# 开发池提案修正：保护历史confirmation患者

状态：metadata-only审计，尚未重分配角色、生成split或训练。

原提案拟将972扫描/444人划分355/89人，同时承诺confirmation保持保护；进一步检查发现两者不能同时按原文实现。`data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/sealed_confirmation/confirmation_cases.jsonl`含108扫描/53人，这53人全部也在972 train manifest里。

因此必须先从原train池排除这53人，剩864扫描/391人；建议固定313 fit/78 development患者，实际scan数批准后生成并冻结。原fixed validation为108扫描/50人，与972 train池患者交集为0。此次读取仅限ID/manifest，不读取confirmation影像、标签数组或性能。

这个修正只明确新方案的训练边界，不能倒推历史全972模型从未见过confirmation，也不能把旧confirmation重新称为盲测。此前文档的355/89提案被本修正取代。

数据角色调整仍须用户明确批准；goal恢复不等于此项授权。
