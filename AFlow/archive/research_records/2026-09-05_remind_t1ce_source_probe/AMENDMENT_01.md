# Amendment 01：官方`.sums`为空MD5，改用双传输字节一致门

状态：2026-09-05 UTC，`COHORT_CANDIDATES.json`与`PROBE_PREACCESS.json`生成后、任何ReMIND DICOM/NRRD内容下载前冻结。病例、首例、资产路径、几何/标签门、成本级别和0模型访问不变。

## 新发现

官方package755的`.sums`有356行、路径集合完整，但356/356 checksum均为`d41d8cd98f00b204e9800998ecf8427e`（空文件MD5），与浏览API中非零NRRD大小明显矛盾。原PREREG“NRRD MD5命中`.sums`”是不可满足且无意义的内容完整性门；`COHORT_CANDIDATES.json`中的`official_md5`字段因此只能标记为`published_but_invalid_empty_md5`，不得拿来验收。

这不是文件下载失败或标签值结果，因为当前尚未请求任何DICOM/NRRD transfer。它是首次内容访问前发现的官方metadata质量问题。

## 修订后的固定完整性门

1. `.sums`只用于冻结356个官方相对路径；不再声称它提供有效内容hash。
2. Probe两份NRRD仍只能请求原冻结的ReMIND-002 manual tumor与automatic cerebrum路径。每份通过TCIA package755独立传输到两个新临时目录；两次bytes和SHA256必须exact，且非空。
3. 对浏览API已给出的ReMIND-002远端size，两个本地文件都必须匹配4,747/1,947,370 bytes；transfer spec source basename/path也必须匹配。Aspera返回码必须为0，token不得落报告。
4. DICOM仍只用冻结SeriesInstanceUID经TCIA NBIA HTTPS获取；PatientID、StudyUID、SeriesUID、SOP实例唯一性及实例数192必须与官方series metadata一致。保存ZIP/逐实例SHA作为本地receipt；官方未给内容hash，不冒充官方hash。
5. 后续几何、二值、非空、tumor⊂cerebrum及无模型规则完全不变。任何双传输字节、size或UID门失败即停止，不换病例或来源。

## 证据边界

双传输exact可排除两次官方传输不一致，但不能替代发布者提供的独立生成时checksum。即使全门通过，来源证据应注明“official transport + repeated local SHA, no valid publisher content checksum”。正式43例若继续，必须再次评估是否接受该较弱完整性等级，不能自动继承probe授权。
