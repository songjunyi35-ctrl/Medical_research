# MOTUM Dataset

This is a public dataset presented at our manuscript titled "A Multi-Center, Multi-Parametric MRI Dataset of Primary and Secondary Brain Tumors" under review in Scientific Data 



## Description

We introduce a multi-center, multi-origin brain tumor MRI (MOTUM) imaging dataset obtained from 67 patients: 29 with high-grade gliomas, 20 with lung metastases, 10 with breast metastases, 2 with gastric metastasis, 4 with ovarian metastasis, and 2 with melanoma metastasis. This dataset includes anonymized digital imaging and communications in medicine (DICOM) files alongside processed FLAIR, T1-weighted, contrast-enhanced T1-weighted, T2-weighted sequences images, segmentation masks of two tumor regions, and clinical data. Our data-sharing initiative is to support the benchmarking of automated tumor segmentation, multi-modal machine learning, and disease differentiation of multi-origin brain tumors in a multi-center setting.


## Data Organization
Multi-contrast MRI, segmentation masks, radiomics features and the acquisition parameters from individuals are stored with the following format. For example, 

`./sub-0001/anat/` stores four imaging sequences of Subject 0001 including FLAIR (`sub-0001_flair.nii.gz`), T1 (`sub-0001_t1.nii.gz`), T2 (`sub-0001_t2.nii.gz`), T1-contrast (`sub-0001_t1ce.nii.gz`) are stored.

`./derivatives/` stores the segmentation masks, radiomics, and meta information of four modalities of all subjects. 

Specifically, for **Subject 0001**, `./derivatives/sub-0001` stores two segmentation masks (`flair_seg_label1.nii.gz` and ` t1ce_seg_label2.nii.gz` derived from FLAIR and T1-ce, respectively), radiomics features (`flair_radiomics.csv`,  `t1_radiomics.csv`,  `t2_radiomics.csv`, `t1ce_radiomics.csv`) and meta information (`flair_param.json`, `t1_param.json`, `t2_param.json`, and `t1ce_param.json`). 


`./Participants.xlsx` stores the meta information of each subject. 

