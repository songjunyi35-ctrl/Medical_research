# Amendment 03: deterministic directory sort key

4并发网络读取完整列出150个目录/879条目后，在写任何输出前因Python尝试直接比较两个dictionary而报TypeError。Package payload、NIfTI/label/model仍为0。

唯一修改：目录条目从无key的`sorted(generator)`改为按冻结metadata字段`path`排序。文件列表本来已按path排序；病例解析、配对、大小和访问门均不变。修复不查看或改变任何像素/结果。
