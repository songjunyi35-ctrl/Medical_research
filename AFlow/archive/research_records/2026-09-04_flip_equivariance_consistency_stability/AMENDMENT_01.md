# Preregistration amendment 01: make the consistency distance zero at equality

Status: frozen after split generation and initial implementation review, before model training, optimizer steps, checkpoint creation, or C/D inner access

## Failure found

The original preregistered soft-Dice expression used denominator `sum(p)+sum(q)`. For nonbinary probabilities it is nonzero even when `p==q`, so it mixes transformation agreement with confidence sharpening. A method described as pure flip-equivariance consistency cannot be cleanly attributed with that formula.

The issue was found by static/unit review before any training. A first train-only V2 scale command completed forward computation but failed while converting the final equality-check vector to a scalar, so it produced no usable metric artifact. The corrected print-only command was rerun on the same64 train-role cases. No C/D inner, fixed validation, outer, confirmation, test270, UCSD75, or other external cohort was accessed.

## Single amendment

Replace only the consistency denominator:

- Original: `1-(2*sum(p*q)+1)/(sum(p)+sum(q)+1)`.
- Amended: `1-(2*sum(p*q)+1)/(sum(p^2)+sum(q^2)+1)`.

The amended squared-denominator soft-Dice distance is exactly zero when the aligned probabilities are identical. Everything else remains frozen, including sigmoid inputs, per-volume then batch mean reduction, seven-reflection cycle, two student forwards in both arms, Control weight0.0, Candidate weight0.1, original supervised/KD loss, seeds, splits, epoch168, gates, and forbidden roles.

## Updated train-only scale evidence

Across the same64 train-role cases and seven nonidentity reflections, the amended distance mean is`0.04784066`, range`0.01144826–0.15794688`, and the exact-equality check is`0.0`. Weight0.1 contributes about`0.00478`, roughly2% beside the observed late training loss around`0.22`; therefore the preregistered0.1 weight remains unchanged and is not re-tuned.

The already generated C/D split remains valid because its selection used only subject scan counts and frozen seeds, not the consistency formula, weight, images, labels, or metrics. All subsequent implementation/checkpoint/report artifacts must bind both the original PREREG SHA and this amendment SHA.

Evidence: [original preregistration](PREREG.md), [original scale audit](TRAIN_ONLY_SCALE_AUDIT.json), [amended scale audit](TRAIN_ONLY_SCALE_AUDIT_V2.json).
