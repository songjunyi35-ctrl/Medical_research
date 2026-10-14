# Implementation amendment 01: nested cache receipt field

首次formal命令在output/checkpoint/model前读取证据收据时，用了不存在的顶层`cache_receipt_sha256`；实际冻结字段是`cache.receipt_sha256`。KeyError后started/raw均不存在，checkpoint未加载，CPTAC forward/metric为0。

唯一修复：evaluator读取嵌套字段；旧MODEL_PREACCESS保留，新建V2绑定修后evaluator。Cohort、source/cache/checkpoint、视图、阈值、bootstrap、Africa-informed解释门和一次性规则不变。
