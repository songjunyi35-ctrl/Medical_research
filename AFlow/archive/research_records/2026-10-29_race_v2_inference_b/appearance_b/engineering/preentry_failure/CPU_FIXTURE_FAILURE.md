# 原CPU fixture失败记录

2026-10-29 18:50前的合成路由测试中，独立single fixture漏了schema要求的candidate_id，触发 final270.py selection 的列表筛选 KeyError: candidate_id。该测试未通过，不能改写为通过。随后在CPU门完成前误提前启动17303/17304，它们因新freeze尚未写入而在verify_entry/verify_transfer处退出；未访问任何新患者/270数组或网络。

修复只补齐合成fixture字段；生产研究core/run和已发布108结果未改。之后独立重新通过CPU门、生成并worker严格自核源码/配方冻结，再恢复17455/17456两个仅等待器。原失败JSON/初始启动收据逐字节保存在本目录。
