# Amendment 001: Gamma Stage A inner evaluator freeze

Status: frozen at paired formal epoch2/168 before completion, preaccess receipt, or inner/validation/test access.

No method, Gamma, data, seed, epoch, metric, gate, or access setting changes.

- Evaluator: `scripts/evaluate_t1ce_wt_gamma_aug_stage_a_inner.py`
- SHA-256: `8fe0eafee645d7519c4827c51a970723a33915d69683c53c4f074dfb20e3dffb`
- Tests SHA-256: `a0bac2af3630be08ebb45aa9908740086608e54b89b979775678060e9a66c48b`
- Gamma utility + evaluator suite: `8 passed`.

Before inner access the evaluator requires exact initial state, sampling weights, all168 sampler sequences, first pre-Gamma image/label,162624 total candidate draws,Gamma applied fraction0.25–0.35,range0.7–1.5 excluding1 at extrema,mean/population-std error<=1e-4,label/global RNG unchanged,and both training/reload/resource/access gates.

Promotion remains: Candidate strictly beats Control Scan/Subject,Point not lower,paired Scan/Subject CI lower bounds>=-0.003,10,000 bootstrap draws seed20261083. Failure prohibits Full972 final and all Gamma tuning.

Access at freeze: train-only inner=false,fixed validation=false,outer=false,confirmation=false,test270=false.
