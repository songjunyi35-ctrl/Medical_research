# WAW术后only-patient31来源审计

状态：✅ 小文件/全mask来源门通过；大postop MRI包、cache、checkpoint、性能未访问。

WAW final描述称161份postop exams；最终metadata/mask包实际各160份、92 patients，160/160 T1c-mask basename严格配对。已揭示preop152患者先按同release patient ID排除：57名重叠，余35名postop-only/54 scans。再要求官方`days_postop`非空，得到31名/48 scans；每人按最小day、再exam ID确定最早一张，冻结31 scans。四名只有空日期的患者与17张后续numeric scans均在像素性能前排除。

- 31人相对WAW preop152 patient overlap=0；days postop 0–22天，其中day0/1为7/17例。
- WHO grade2/3/4=`1/1/29`，直接偏向当前高等级术后任务。
- 31 masks均240×240×155、1mm³，affine与T1c metadata exact；raw label case presence 0/1/2/3/4=`31/12/31/25/30`。
- WT固定raw `{1,2,3}`、label4切除腔为背景；31/31 WT非空，native WT min/median/max=`13,253/67,864/204,337`。
- Zenodo小文件官方MD5 exact。大包仍冻结为`2_2_postop_scans_coreg.zip`，3,321,955,416 bytes、官方MD5 `ab9da3eb2a3aff73b8f3b95e01508c1c`。
- 同一Warsaw source但患者与preop已见队列严格不同；相对BraTS2024七站institution-disjoint、无跨库patient bridge。Public、非hidden blind。

决定：可在独立七问/预注册下做一次冻结面板。大包必须官方size/MD5/CRC exact；只提取31冻结T1c。不得加入日期空患者、换后续时间点或把57名preop已见患者混回。

证据：[机器审计](SOURCE_AUDIT.json) SHA `ca7d67892e42ae4d193f87ebdd983fb79314091b84d2e448021a71da18e0fe3d`、[冻结cohort](FROZEN_COHORT.json) SHA `965b2fe0f331d782dbb6bb136d6089c1181419a9a75dbd8f198cbeff33284180`、[preop来源](../2026-09-06_waw_gliomai_source_audit/RESULT.md)。
