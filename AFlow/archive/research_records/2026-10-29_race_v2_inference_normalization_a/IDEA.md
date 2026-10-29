# Seven prospective questions

1. Problem: whole-brain moment normalization can depend on abnormal tissue intensity. Can a conservative seed-excluded reference improve this fixed RACE model?
2. Evidence: WhiteStripe motivates healthy tissue references; it is not this method and does not guarantee tumor segmentation benefit. Primary preprocessing research also reports null/harmful effects.
3. Difference: old fixed gamma/gain/offset, geometric ROI/phase and output covariance do not compute per-image affine coefficients from a preceding lesion seed.
4. Mechanism: original identity seed excludes suspected disease with a fixed3-voxel Chebyshev dilation. Remaining nonzero image voxels estimate mean/SD. Bounded positive affine and partial strength retain anatomy, contrast ordering and original reference support.
5. Test: exactly4 inputs x2 output seed weights=8 new configurations against equal8, fixed patient5fold and patient-cluster10000 seed20261029 intervals; preserve all worsening cases and primary final-score Point.
6. Failure risks: seed omissions contaminate healthy reference; false-positive seed excludes normal tissue; nonlesion brain is not white matter; frozen training normalization may already be adequate; replacing7 original views can lose robustness. Fallback and bounds are prospective, not GT routing.
7. Budget/deployment: new108 neural forwards<=3024, existing seed cached; deployment1 original seed+7 adapted mirrors=8. One family, no training, no270 access, same geometry and score, <=30min. Same inputs alias cached predictions rather than repeat neural work.
