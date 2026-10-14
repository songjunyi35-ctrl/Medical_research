"""Same canonical numerical preprocessing, retaining external subject identifiers."""
from sammed3d.t1ce_binary import (
    _shape, _spacing, _normalize_nonzero_t1ce, _resize, model_spacing_mm,
    residual_tumor_target, WholeVolumeResizeGeometry, T1ceSupervisedView,
    T1ceBinaryDataError, T1ceBinaryGeometryError)


def prepare_external_t1ce_arrays(case_id, image, labels, spacing_mm, *, output_shape=(128,128,128)):
    if not isinstance(case_id,str) or not case_id.startswith('sub-') or not case_id.endswith('-0'):
        raise T1ceBinaryDataError('expected original OpenNeuro subject id plus scan suffix -0')
    target_shape=_shape(output_shape,'output_shape')
    normalized,brain_mask,stats=_normalize_nonzero_t1ce(image)
    source_shape=_shape(normalized.shape,'T1ce shape')
    resized_mask=_resize(brain_mask,target_shape,'nearest-exact').bool()
    resized_image=_resize(normalized,target_shape,'trilinear')
    resized_image.masked_fill_(~resized_mask,0.)
    geometry=WholeVolumeResizeGeometry(original_shape=source_shape,model_shape=target_shape,
        original_spacing_mm=_spacing(spacing_mm),model_spacing_mm=model_spacing_mm(source_shape,spacing_mm,target_shape))
    target=residual_tumor_target(labels)
    if tuple(target.shape)!=source_shape:
        raise T1ceBinaryGeometryError('external image and label shapes differ')
    resized_target=_resize(target,target_shape,'nearest-exact').bool()
    return T1ceSupervisedView(case_id=case_id,image=resized_image.unsqueeze(0).contiguous(),
        target=resized_target.unsqueeze(0).contiguous(),source_target=target.unsqueeze(0).contiguous(),
        brain_mask=resized_mask.contiguous(),normalization=stats,geometry=geometry)
