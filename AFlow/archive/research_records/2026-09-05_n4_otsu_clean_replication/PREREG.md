# Frozen N4-Otsu200 mask-only protocol

Status: frozen before Otsu implementation/mechanism/cache/student or first E/F readout.

- The plain-language IDEA is the normative complete protocol. No free parameter other than the categorical mask policy is searched.
- Shared source972 SHA256:6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65.
- Shared E/F SPLIT.json SHA256:68975292cc10ab581657a97789050a28cc03d4a3bc3b0991682b7284bfa9c3f2.
- Shared canonical index SHA256:7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd; labels SHA256:bf470d48a5f115f09407ad351bb6dee11ee36bdd0c2a4285ae0bd9cc22125724.
- Shared no-N4 IDEA/PREREG hashes:6ffe103b4c592850eac4f4f214c157bf603e97a8e5a436fec926401cb7a41921 / 1dbd9376d682f2893c0a99c5656995cbb9fe02ba6a5fca239251954e4fc7a519. Its AMENDMENT_01 engineering-label access clarification also applies. Historical hash-bound files will not be modified.
- Otsu mask exactly SimpleITK.OtsuThreshold(full_resolution_float32_image,0,1,200); shrink mask and image by2 for N4; iterations[50,50,50,50], all other defaults unchanged; correct original full image by exp(log bias field), then zero only originally nonpositive values. No mask erosion, percentiles, clipping, new intensity unit, or alternative N4 implementation.
- Mechanism cohort: same first3 train case_ids in source-sorted order. Original float32 x and encoded float32(x/sigma−mu/sigma); decode by subtracting exact border-background value in float64 then castfloat32, as prior audited formula. Compare two Otsu-N4 pipelines against each other, never claim either equals historical positive-mask N4. max≤0.001,mean≤0.00001 and full support/bbox exact for all3. Any failure closes before any candidate cache/student/model readout.
- Candidate students E/F seeds20261313/20261323,4.70M width16,168 epochs,968 effective draws/121 steps,batch8,原AdamW/LR/augmentation/DiceCE+KD identical to shared canonical students. Samefold newly prepared history/EMA teachers are fixed at0.75/0.25 logits, KD weight0.3/T2. Teacher checkpoint fit SHA must match ownfold930; no historical756 teacher reuse.
- Implementation, mechanism receipt, candidate cache, tests and smoke hashes will freeze before student formal launch. Both candidates and all shared models must complete or fail their preregistered pre-evaluation engineering gates before either method opens E/F.
- E/F paired metrics can reuse saved canonical control rows from the once-only no-N4 family readout after all models freeze. Otsu candidates each require42×8 forwards=336; no extra control forward. Same ordered8 reflection list, float32 logit mean,threshold>0,argmax point; no SAM/postprocessing.
- Perfold/style gates and patient-cluster10,000 bootstrap are exactly IDEA; allprimary directions/CI/point/mean gates must pass. Finite/reload/initial/fit/sampler/augmentation-RNG/first-label parity and ≤24GiB/≤6h/student gates must also pass.
- Old validation/outer/confirmation/test270/UCSD75 model access forbidden. NewRHUH source labels/models remain closed. No result authorizes old-cohort retuning or revising this mask formula.
- Archive positives/negatives and access flags. If mechanism fails, do not delay the originalno-N4 family readout once its own models finish; mark this branch closed and remove it only from the pending model barrier, never from the idea history.
