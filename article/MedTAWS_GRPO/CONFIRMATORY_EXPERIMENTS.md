# Confirmatory Experiments Required Before Submission

The manuscript is structurally complete, but the following evidence must be produced before it can support a SOTA or generalization claim.

## Frozen Task Contract

- Input: T1ce only, with exact source suffix and preprocessing hash recorded.
- Target: choose and freeze either raw RC `{4}` or residual-tumor union `{1,2,3}`. Do not mix them.
- Interaction: three accumulated clicks; first positive point from foreground, later positive/negative points from FN/FP error regions; recurrent mask enabled.
- Split: subject-disjoint train/validation/test. Longitudinal scans from one subject stay in one partition.
- Metrics: subject-balanced Dice and IoU, HD95, per-click Dice C1/C2/C3, best-of-three only as a secondary interactive metric.
- Model selection: validation only. Test remains sealed until every row below is frozen.

## Required Four-Arm Table

| Arm | Weights | Workflow | Status |
|---|---|---|---|
| A | Frozen SAM-Med3D base | Frozen official three-click workflow | Not run under final contract |
| B | Frozen SAM-Med3D base | MedTAWS-selected workflow | Not run |
| C | GRPO-style adapted | Same frozen official workflow as A | Not run |
| D | Same adapted checkpoint as C | MedTAWS-selected workflow | Not run |

## Search Attribution

Compare MedTAWS, score-softmax, greedy, and uniform parent policies with the same root, proposal model, operator set, validation cases, number of workflow evaluations, SAM-Med3D forwards, and LLM calls. Run at least three search seeds. Report final best Dice, best-so-far AUC, valid-workflow rate, model forwards, LLM calls, wall time, and peak memory.

## Training Attribution

Compare supervised Dice-CE and GRPO-style alignment with the same initialization, train subjects, validation subjects, epochs, optimizer, learning rate, prompt trajectories, and checkpoint-selection rule. Run at least three seeds. Include K=2 versus K=4 and anchor 0.10 versus 0.05. Do not select a final configuration using test results.

## One-Time Holdout

Freeze checkpoint hashes, workflow hashes, target mapping, preprocessing, click simulator, thresholds, metrics, code revision, and statistical plan. Then evaluate the four arms once. Report paired subject-level confidence intervals and a paired permutation or Wilcoxon test. A failed SOTA comparison is still reported; the test must not be reopened for tuning.
