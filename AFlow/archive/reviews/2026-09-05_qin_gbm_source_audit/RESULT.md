# QIN GBM Treatment Response来源审计：无体素肿瘤真值

日期：2026-09-05 UTC

结论：不进入当前T1ce/Whole Tumor外部确认。TCIA官方Version 2提供54名新诊断GBM患者术后、治疗前相隔2–5天的double-baseline MRI，包含T1 pre/postcontrast、T2、FLAIR、MEMPRAGE及扩散/灌注序列；但collection Data Types与下载表只有MR Images，没有SEG、RTSTRUCT、ROI或专家体素肿瘤标注，无法计算当前Scan/Subject Dice与Point hit。

官方T1 structural protocol为2D FLASH、256×216、仅23 slices、5mm厚；即使未来另有临床ROI，其栅格也与当前3D WT任务有显著caveat。该数据的官方用途是ADC/Ktrans/rCBV test-retest，不是肿瘤分割benchmark。

本审计只读取TCIA官方网页；没有下载33.51GB/589,314 images、manifest、DICOM或任何模型输出。关闭“QIN-GBM作为当前带WT真值外部队列”路线，不把无标签数据或test-retest参数代理成分割真值。它未来可用于无标签输入稳定性研究，但那需要新的目标与协议。

一手来源：[TCIA QIN GBM Treatment Response](https://www.cancerimagingarchive.net/collection/qin-gbm-treatment-response/)、[DOI 10.7937/K9/TCIA.2016.nQF4gpn2](https://doi.org/10.7937/K9/TCIA.2016.nQF4gpn2)。
