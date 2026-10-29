# RACE seed-excluded healthy-support normalization

Eight fixed new configurations completed on108 validation scans. Best overall `equal8_logit_t05` Scan 0.768011881. Best new `healthy_q0.1_alpha0.5_seed0.125` Scan 0.767943453, Subject 0.758179505, Point 98/108. Delta versus equal8 -0.000068428, descriptive patient95% interval [-0.00015599800552610157, 1.551642746311945e-05].

Deployment uses one original seed and seven normalized mirrors, eight forwards total. New validation neural forwards 3024/3024; all eight decoders reuse cached logits. No training or new270 scoring. Validation and historical test reuse remain test-tuned; no independent confirmation. Whole original crop and native1mm score unchanged. Full counts, masks, all worsening cases, patient folds, cost and exact recipes are retained in JSON.
