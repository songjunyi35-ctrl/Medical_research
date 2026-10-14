# UTSW-Glioma 官方来源审计结果

## 结论

UTSW-Glioma 是合法且可精确枚举的新外部来源候选，但目前只完成来源身份与文件清单，不授权模型评测。TCIA官方TSV有625个唯一术前病例，其中362例明确标为人工细化、263例仅有自动标签；UT Southwestern不在BraTS 2024 post-treatment的七个贡献机构中。它未来必须作为“跨机构且跨治疗阶段”的独立队列，不能与现有post-treatment test270混排。

## 官方身份与完整清单

- TCIA package ID：1265；许可：CC BY 4.0。
- TSV：625/625唯一Subject ID，362 `Yes`、263 `No`；SHA-256 `f7ff889fa7d197b2f589695d3e7ea8a2d1d198c4f9c0044fc71898a08aa24ed0`。
- Faspex完整iteration-token遍历：625病例目录、6349文件、24,667,030,300 bytes。
- 文件集合逐例零偏差：每个自动-only病例固定9文件，每个人工病例固定11文件；`362×11+263×9=6349`。
- 完整清单 SHA-256：`5f210c0a1e0c33f7b71a56d9bbce8b52a62f24e2a0b53655aa3a098dbde3f871`；总结报告 SHA-256：`f116b23a116818e3d22618b2a15ab4baeb2794ea45248cff68e3647f4d3fa9af`。
- 6349项均为指向正大小文件的symbolic link；官方目录没有逐文件checksum。

人工子集画像：362例中glioblastoma 226例；grade 4/3/2/NA为236/63/61/2；Siemens/GE/Philips/Hitachi为168/107/80/7。若未来另行解决输入版本并只取主`brain_t1ce`与全网格`r`人工标签，目录总大小约1,813,086,599 bytes；ANTs T1ce加同标签约1,758,502,010 bytes。数字只用于下载/人群规划，不授权或选择模型。

## 必须保留的实现失败

初版脚本把`/page`返回的前100项误当完整目录，又预设T1GD/唯一segmentation命名，得到1076文件和362例全不匹配。IBM官方文档与TCIA当前前端随后证明下一页token位于`x-aspera-next-iteration-token`响应头、通过query传回；固定探针还发现真实文件名为`brain_t1ce`且人工病例有两份manual-correction标签。初版`SOURCE_AUDIT.json`及失败解释保留，不覆盖。

Amendment 01首次正式全量执行又因12并发TLS `WRONG_VERSION_NUMBER`中止且没有报告/清单落盘；Amendment 02只把并发降至4并加入3次连接重试，随后完整通过。传输实现失败不冒充来源失败，也没有访问像素。

## 尚未解决的边界

每个人工病例同时包含两份T1ce和两份manual-correction标签，目录层无法确定唯一image-label配对。因此来源审计只把362例固定为候选；后续固定3例桥接已另行预注册，不能根据模型分数挑文件。

## 访问与成本

本来源审计仅访问网页、96KB TSV和Faspex目录元数据；0影像体素、0标签体素、0模型、0旧cohort、0GPU。全量22.9GB页面标称包未下载。

## 证据

- [计划](PLAN.md)
- [初版失败机器报告](SOURCE_AUDIT.json)
- [分页与命名修订](AMENDMENT_01.md)
- [传输稳定性修订](AMENDMENT_02.md)
- [首次全量执行失败收据](EXECUTION_01_FAILURE.json)
- [完整清单机器报告](SOURCE_AUDIT_AMENDMENT_01.json)
- [完整官方目录](OFFICIAL_INVENTORY_AMENDMENT_01.json)
