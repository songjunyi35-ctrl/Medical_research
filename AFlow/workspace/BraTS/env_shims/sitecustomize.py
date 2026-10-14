"""Environment-only compatibility shim for torch >= 2.6 checkpoint loading.

Why this exists
---------------
`scripts/train_sammed3d_t1ce_wt_partial_unfreeze.py` stores `args.output_dir`
as a `pathlib.PosixPath` inside the checkpoint payload. PyTorch 2.6 flipped the
default of `torch.load(..., weights_only)` from False to True, and the safe
unpickler rejects `pathlib.PosixPath`. The shared loader
`sammed3d.peft.load_model_with_adapters` calls `torch.load` without an explicit
`weights_only`, so strict reload fails on this host (torch 2.7.1) while it
succeeded on the original A10 worker (torch < 2.6).

This is purely a deserialization-permission difference. It cannot change any
tensor value, hyperparameter, seed, split, or architecture.

Why a shim instead of a source edit
-----------------------------------
`sammed3d/peft.py` is byte-pinned by the already-completed rank-0 matched arm
(`scripts/preflight_sammed3d_t1ce_rank0_matched.py` expects peft_module SHA-256
`4844b1a086c74737f786399c4c67a2b3c56ae34f35b152e423f63591f226bc63`, and
`scripts/summarize_sammed3d_pair.py` repeats it). Editing the file would break a
frozen fail-closed guard, and rewriting the expected hash to make a run start is
explicitly forbidden by the handoff. This shim keeps every pinned artifact
byte-identical.

It is also strictly safer than `weights_only=False`: the safe unpickler stays
enabled and exactly one benign stdlib class is allowlisted.

Activation is explicit and opt-in, never implicit:
    PYTHONPATH=workspace/BraTS/env_shims bash scripts/<frozen launcher>.sh
"""

import pathlib

try:
    import torch.serialization

    torch.serialization.add_safe_globals([pathlib.PosixPath])
except Exception:  # pragma: no cover - never block interpreter startup
    pass
