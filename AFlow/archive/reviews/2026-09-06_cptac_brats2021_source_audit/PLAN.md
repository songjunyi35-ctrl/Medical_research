# CPTAC-GBM/BraTS2021公开NIfTI来源审计计划

状态：冻结在`.sums`、患者payload、NIfTI数组、标签值和模型访问前。

目标：从TCIA RSNA-ASNR-MICCAI-BraTS-2021 package636中，仅审计官方映射表标为`CPTAC-GBM`的39位患者（public Training33/Validation6）。检查每例是否有唯一T1CE与人工BraTS seg，能否作为Africa95之后的独立、Africa-informed外部确认。

固定：官方analysis DOI `10.7937/JC8X-9874`、package636、`BraTS2021_MappingToTCIA.xlsx` SHA-256 `223243c5821c282e90da980ba38d860ab2d2a597d1f4f1800ebe227329a2dc0b`。只列Training/Validation的CPTAC NIfTI目录，不列DICOM主体或其他1,441例。

门：39 PatientID/BraTS ID唯一；33/6角色exact；39目录exact；每例5个正大小NIfTI且`t1ce/seg`各1；无hidden test。失败在payload前停止。通过只允许冻结来源cohort，任何下载/模型仍需新的七问/PREREG。

一手来源：[TCIA analysis result](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/)、[CPTAC-GBM collection](https://www.cancerimagingarchive.net/collection/cptac-gbm/)。
