# True-128³ late-cosine EMA preregistration

This experiment returns to the current efficient 128³ plain locator and adds only an exponential moving average of weights. EMA is initialized as an exact copy of raw weights after epoch100. From epoch101 through epoch200, every optimizer step applies `ema = 0.999 * ema + 0.001 * raw`; integer buffers are copied. The decay and start epoch are not searched.

All 756 train cases / 346 subjects are used. Model, cache, sampler, augmentation, loss, AdamW configuration, batch8, and the 200-epoch constant-plus-cosine schedule remain fixed. Raw and EMA checkpoints are saved at epochs168 and200 without validation access. EMA does not add inference parameters, FLOPs, or memory relative to the raw model.

The primary development comparison is fixed epoch200 EMA versus same-trajectory raw epoch200 on the historical fixed 108-case/45-subject validation set. All gates must pass: positive scan and subject Dice gains, point-hit drop no worse than `-0.01`, paired subject 95% CI lower bound at least `-0.003`, and exact raw/EMA reloads. Epoch168 is secondary. Failure ends the EMA direction without decay, start-epoch, threshold, or test-based tuning.

The already revealed resolution test result is not used to choose EMA hyperparameters. Test270 remains closed during this training/validation stage.
