# Evidence Matrix

This file records the claim boundary for every numerical result used by the manuscript.

| Claim or table | Source evidence | Allowed wording | Prohibited wording |
|---|---|---|---|
| GRPO-style 118/e10 mean Dice 0.58932 | `AFLOW_BRATS_HANDOFF.md`, Section 40 | Small two-seed advantage over matched supervised training in a retrospective development split | Subject-disjoint, test, residual-tumor, fully automatic, or SOTA result |
| Matched supervised mean 0.57283 | Section 40 clean rerun | Same data size, epochs, seeds, initialization class, and reload audit | Proof that supervision is generally inferior |
| Mean delta +0.01649 | Per-seed deltas +0.02961 and +0.00337 | Modest historical development signal | Significant or robust clinical improvement |
| 32/64/96/118 progression | Handoff Sections 16, 22, 23, 26, 27 | Development trend as data and/or epochs increase | Single-variable scaling law |
| Group size and anchor ablation | Handoff Sections 16, 18, 20 | K=4 and anchor=0.05 are best among tested two-seed small runs | Universal optimal hyperparameters |
| ET and TC reversals | Handoff Sections 34 and 35 | RC benefit does not transfer under the tested settings | Multi-target improvement |
| Legacy MedTAWS candidate gains | Work #1 completion audit | End-to-end executable search produced target-specific development candidates | Equal-budget superiority over greedy/random or independent-test gain |
| Four-token MedTAWS tier-1 values | Handoff interface retraction | Mention only as a withdrawn diagnostic | Any performance evidence |

## Protocol Facts That Must Remain Visible

- Historical input: single T1 contrast-enhanced channel.
- Historical target: raw resection-cavity label 4, not the current residual union `{1,2,3}`.
- Historical geometry: ground-truth-centered `128^3` crop/pad.
- Historical interaction: five simulated GT-error corrective clicks with recurrent mask guidance.
- Historical population: 200 scans from 39 longitudinal subjects; all 16 validation subjects overlap training.
- Historical aggregation: 16 scans times four trajectories, averaged per rollout rather than per subject.
- No subject-clean 1,350-scan GRPO-style or combined MedTAWS-GRPO result exists yet.
- No sealed-test or external-leaderboard claim is authorized.

## Method-Naming Boundary

- Use **MedTAWS (Medical Tree-guided Adaptive Workflow Search)**.
- State that MedTAWS is built on AFlow.
- Use **stochastic UCB-guided workflow-tree search**, not standard MCTS.
- Use **GRPO-style group-relative prompt-trajectory alignment**, not standard PPO/GRPO.
- Explain that the deterministic threshold action and same-forward detached old log-probability make the forward importance ratio one.
