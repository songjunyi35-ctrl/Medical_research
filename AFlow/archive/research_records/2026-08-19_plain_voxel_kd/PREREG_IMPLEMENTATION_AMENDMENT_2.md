# Implementation amendment 2

The clean V2 control completed all 32 epochs, saved all four fixed checkpoints, performed exact final reload validation, and atomically wrote `TRAIN_DONE.json`. It then exited nonzero only while printing the already-written final summary because the V1 module omitted `import json`.

V3 injects Python's standard `json` module for that final print. It changes no training state or result. The completed control is retained; only the not-yet-run KD arm uses V3.
