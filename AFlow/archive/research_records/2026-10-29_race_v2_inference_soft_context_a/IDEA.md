# Seven questions

1. Problem: most paired-model errors are shared or outside the existing logits convex hull; output reweighting has limited scope. Test an input intervention that may change predictions, not a claim of new patient information.
2. Evidence: trained feature attention motivates suppressing irrelevant context but does not establish this frozen RACE input intervention. It is a weak analogy.
3. History: older ROI crops/resizes and stitches probabilities; phase shifts padded inputs; fixed/global photometric groups and healthy moments are different. Older ROI grids were negative relative to the old final recipe, so success is not assumed.
4. Mechanism: original identity seed defines a smooth distance gain. Preserve the original grid, every original nonzero support voxel, and reference matrix; symmetrically protect contralateral context. No image interpolation or renormalization.
5. Falsification: exact r10/20 x floor0.75/0.9, four configurations only, fixed output equal weights, whole original GT crop, final-score Point, equal8 reference; five patient folds and10000 patient bootstrap seed20261029; retain all worsening cases.
6. Risks: omitted lesions lose contrast; false positives preserve irrelevant context; InstanceNorm changes can propagate globally; bilateral protection may leave too little intervention. Nonzero input does not guarantee preservation of undetected lesions.
7. Budget: original seed1+adapted7=8 deployment forwards, at most3024 new validation forwards, <20minutes; fields cached once per radius and shared across floors. No training, extra model or270 access. Gate failure stops rather than changing parameters.
