# ROI启动预算遍历与已授权清理竞态

新rolling cleanup安全删除已完core直方图时，旧冻结roi_render_v1仍用inline rglob+stat大小求和，遇到恰好被删除的派生npy，导致最后一组ROI在患者模型前退出；不是患者推理数值失败。工程恢复计数现3/6。保留原源码/失败/原任务，roi_render_v2只把inline预算扫描改为guard_v2.stable_size，同样300GiB上限，数值/几何/数据/视图/随机种子不变；从未完任务接续。其他旧冻结数值文件不改，后续新renderer必须使用robust预算统计。无RACE影响。
