# Historical-best + EMA168 logit ensemble preregistration

The two fixed true-128³ epoch168 models are combined on the model grid as `(1-w) * historical + w * EMA168`. Candidate EMA weights are exactly `0.25`, `0.5`, and `0.75`; endpoints `0` and `1` are reported as references. The validation-selected weight maximizes scan Dice with subject Dice as tie-break. Threshold remains raw blended logit `>0`, and the automatic point is the blended-logit argmax. No TTA or postprocessing is allowed.

Promotion requires positive scan and subject gains versus the historical model, point-hit decline no worse than `-0.01`, and paired subject 95% CI lower bound at least `-0.003`. Failure ends this ensemble direction without test-based weight tuning. The method doubles locator inference compute but does not require training or SAM.
