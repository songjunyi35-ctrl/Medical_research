# Pre-implementation amendment 002: canonical RAS before applying crop geometry

Status: frozen after the second train-only artifact attempt failed its exact gate, before any model implementation, GPU, formal, or validation access.

The second builder attempt used the corrected image-only crop definition but applied the frozen canonical crop coordinates directly to the stored NIfTI array orientation. It refused output because23/756 source counts differed and the control weight max-abs difference was`0.0197692`. No artifact or model/evaluation access occurred.

Read-only diagnosis confirmed:

- The old972 and historical formal756 crop geometries are identical for every overlapping case.
- All756 source label headers are stored in LAS orientation.
- The historical cache builder calls `nib.as_closest_canonical` and therefore defines origin/extent in canonical RAS array coordinates.
- Applying those coordinates before canonicalization can alter only tumors touching the image-derived crop boundary, explaining why most cases happened to match and23 did not.

The builder is therefore frozen to call `nib.as_closest_canonical(nib.load(label_path))` before integer checks, shape checks and crop slicing. This operation only permutes/flips the native1mm grid; it does not interpolate or change full-FOV label counts. The artifact records stored and canonical orientation codes for audit.

The exact gates remain unchanged: all756 crop-native WT counts and the full float64 control weight tensor must equal the historical formal dataset exactly. This is a coordinate-system bug fix required to implement the already frozen historical sampling semantics; no sampling formula, arm, model, loss, draw count, epoch, validation or test choice changes.

