# WAW-GliomAI术前来源审计

状态：来源/许可/metadata/全152 masks已审；大MRI包、checkpoint、模型性能均未访问。

## 结论

WAW-GliomAI final Zenodo record `22205631`是当前可立即运行的新公开人工标签来源：open access、CC BY 4.0，来自Medical University of Warsaw。发布说明写156份术前检查，但最终文件本身只有152份coreg四模态metadata和152份术前mask；二者T1c-mask精确交集为152名患者、每人exam0一份。冻结最终文件交集，不寻找或补齐描述中的4份差额。

- 152/152有`*_brain_t1c.nii.gz` metadata，均同时有T1n/T2f/T2w；152/152 mask basename与exam精确配对。
- 全部mask为`240×240×155`、`1 mm³`，affine与各自T1c metadata严格一致；非空WT体素min/median/max=`2,959/65,555.5/269,656`。
- 全局原始label union=`{0,1,2,3}`；0/1/2/3分别见152/84/152/123例。术前没有resection-bed label4，因此本项目WT可无猜测地定义为所有正类。
- WHO grade2/3/4=`44/34/74`；3T/1.5T/0.38T/缺失=`114/33/1/4`，不是单一扫描器同质样本。
- 官方说明：FL-PoST/FeTS给初始分割，最终由神经肿瘤影像专家逐例复核和人工修改。技术验证包含15个术前双读者案例，标签2 FLAIR异常的median Dice为`0.981`，标签1/3分别`0.897/0.927`；样本小，只作标注质量证据。
- 华沙医科大学不在BraTS2024七个来源站，故只称institution-disjoint/no-known-overlap；没有跨库患者ID桥，不称ID证明的patient-disjoint。
- 六个小文件官方MD5全部吻合。待下载影像包冻结为3,164,137,716 bytes、MD5 `0d24a3a2c76bc6fa9b5b6468679676b1`。

## 决定与边界

来源门通过，可在独立七问/预注册下对WAW152做一次冻结checkpoint评测。必须先核大包官方size/MD5、再核152个T1c与mask原生geometry；任一不符即模型前停止。不得使用raw/postop影像、临床字段筛选、缺失描述补例或模型结果排例。

## 证据

- [机器来源审计](SOURCE_AUDIT.json)，SHA-256 `62503b3e680f0fc6922f20d6732a608d226429357bb8e5bb52da67e7ade7d3e7`
- [冻结152病例](FROZEN_COHORT.json)，SHA-256 `25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a`
- [Zenodo final record](https://zenodo.org/records/22205631)
- 审计器 `scripts/audit_waw_gliomai_source.py` SHA-256 `d4a1ad1dcca61bc75b506fee2fc8846435fe81bf6eee1bdb33e2367728193197`

机器审计中的`coreg_image_archive_downloaded=false`、`checkpoint_loaded=false`、`model_forwards=0`保持为来源阶段边界。
