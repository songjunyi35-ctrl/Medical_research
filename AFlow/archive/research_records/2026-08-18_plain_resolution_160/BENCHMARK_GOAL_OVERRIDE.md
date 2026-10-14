# User-directed benchmark-goal override

The strict paired 128³/160³ inner-development protocol was stopped during the 128³ control at approximately epoch60, before any inner-tune tensor iteration. The user clarified that the objective is to maximize the eventual benchmark score, not to estimate a clean resolution-only effect.

The replacement run uses all 756 cases / 346 subjects in the authorized historical train cohort, true-160³ raw-NIfTI preprocessing, batch 8, seed `20260829`, and the complete 200-epoch schedule. It saves fixed epoch168 and epoch200 checkpoints without validation-based selection. Outer selector, confirmation, and the 270-case test remain inaccessible under the standing restriction, so this run produces checkpoints but no benchmark estimate.

The interrupted 648-fit 128³ output is retained as an incomplete engineering record and is not a result.
