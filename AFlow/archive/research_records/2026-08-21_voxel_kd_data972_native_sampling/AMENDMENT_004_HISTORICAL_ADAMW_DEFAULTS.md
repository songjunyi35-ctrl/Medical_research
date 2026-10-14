# Pre-formal amendment 004: restore historical AdamW execution defaults

Status: frozen after initial synthetic/GPU smoke audit and before formal training or validation access.

The original successful voxel-KD runner constructs `torch.optim.AdamW(student.parameters(), lr=..., weight_decay=...)` without specifying `foreach` or `fused`. In the frozen environment this records `foreach=None` and `fused=None`, allowing PyTorch's historical default implementation selection. The inherited fixed-compute data runner explicitly passed `foreach=False,fused=False`.

Although both new arms would remain mutually matched, retaining explicit false could prevent `native756_control` from reproducing the original successful optimization semantics. The native runner must therefore override only those inherited arguments back to `foreach=None,fused=None`. AdamW betas, epsilon, amsgrad, maximize, capturable, learning-rate schedule, weight decay, parameter order, zero-grad, gradient clipping and every method/data setting remain unchanged.

Both GPU smokes must be rerun from scratch after this fix and the implementation receipt must supersede the earlier smoke hashes. No formal checkpoint or validation/test access exists at the time of this amendment.

