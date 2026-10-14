# Implementation amendment 1

V1 trained the control through global epoch176 but failed before its first checkpoint save because the CLI namespace omitted the already-frozen architecture metadata fields `init_filters` and `dropout`. No checkpoint, validation metric, or test access occurred.

V2 adds only fixed CLI defaults `init_filters=16` and `dropout=0.1`, matching the loaded historical architecture. Data, initialization, RNG seed, sampler, augmentation, optimizer, LR, continuation epochs, KD formula, temperature, KD weight, checkpoints, validation selection, and gates are unchanged. The incomplete V1 output is not a result and is replaced by a clean rerun from historical epoch168.
