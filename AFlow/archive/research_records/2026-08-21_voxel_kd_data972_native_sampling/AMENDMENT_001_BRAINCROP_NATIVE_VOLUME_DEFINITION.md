# Pre-implementation amendment 001: historical native-volume means image-crop-native volume

Status: frozen after the first train-only artifact attempt failed its exact gate, and before any model implementation, GPU, formal, or validation access.

## Trigger

The first builder attempt defined native WT volume over the entire source NIfTI. It processed all972 train labels but refused to write the output because one of the overlapping756 cases differed from the historical formal index and the full weight tensor was not exact. No artifact, model training, validation, outer, confirmation, or test access occurred.

Read-only diagnosis found exactly one mismatch:

- Case `BraTS-GLI-02660-100`.
- Full source NIfTI labels `{1,2,3}`: `115183` voxels.
- Historical formal index source-positive count: `115179`.
- The case contains label4, but label4 is not the four-voxel discrepancy.
- Historical cache builder computes `residual_tumor_target(label)` only after applying the image-derived nonzero brain bounding box. Four `{1,2,3}` voxels lie outside that box.

## Corrected frozen definition

The intended experiment is to restore the historical successful sampling semantics, not invent a new full-FOV count. Therefore `native_volume_i` is now frozen as:

1. Load the source train label NIfTI without resampling.
2. Read the existing 972 cache `crop_geometry[case_id].origin/extent`, which was derived from T1ce image intensity only and does not use GT.
3. Slice the source label with that frozen box.
4. Count labels `{1,2,3}` inside the crop; exclude0 and4.

The candidate cache index is frozen at SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`. Every case must have matching `native_shape`, positive extents and in-bounds geometry. The artifact records both full-FOV WT count and the crop-native WT count for audit, but only the crop-native count enters sampling.

The overlapping756 crop-native counts and the derived float64 control weights must still be exactly equal to the historical formal index/dataset. Any remaining mismatch stops the method. This amendment changes only the pre-implementation data-definition bug exposed by the exact gate; it does not change the weight formula, arm populations, draw count, model, loss, epoch grid, promotion gates, or any observed validation/test result.

