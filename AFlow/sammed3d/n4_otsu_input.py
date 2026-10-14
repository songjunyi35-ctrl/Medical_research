"""N4 fitting-mask-only variant: official SimpleITK Otsu200 example."""
import hashlib
import numpy as np
import SimpleITK as sitk


def n4_otsu_correct(volume):
    data=np.ascontiguousarray(volume,dtype=np.float32)
    if data.ndim!=3 or not np.isfinite(data).all():raise ValueError("finite3D input required")
    image=sitk.GetImageFromArray(data)
    mask=sitk.Cast(sitk.OtsuThreshold(image,0,1,200),sitk.sitkUInt8)
    mask_array=sitk.GetArrayFromImage(mask)
    if not np.any(mask_array) or np.any((mask_array!=0)&(data<=0)):
        raise ValueError("Otsu fitting mask empty or includes nonpositive input")
    corrector=sitk.N4BiasFieldCorrectionImageFilter();corrector.SetMaximumNumberOfIterations([50]*4)
    corrector.Execute(sitk.Cast(sitk.Shrink(image,[2]*3),sitk.sitkFloat32),sitk.Shrink(mask,[2]*3))
    log_bias=corrector.GetLogBiasFieldAsImage(image)
    result=np.asarray(sitk.GetArrayFromImage(sitk.Cast(image,sitk.sitkFloat32)/sitk.Exp(log_bias)),dtype=np.float32)
    result[data<=0]=0
    if not np.isfinite(result).all() or not np.array_equal(result>0,data>0):raise ValueError("corrected full support changed")
    return result,{"fit_mask_sha256":hashlib.sha256(mask_array.tobytes()).hexdigest(),"fit_voxels":int(mask_array.sum()),
                   "full_positive_voxels":int((data>0).sum()),"mask_policy":"OtsuThreshold(full_image,0,1,200)","mask_used_only_for_fitting":True}
