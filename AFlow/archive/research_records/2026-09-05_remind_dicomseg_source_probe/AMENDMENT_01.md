# Amendment 01：同一pydicom wheel解压使用，不安装

状态：2026-09-05 UTC，在任何DICOM SEG下载/数组解码前冻结。病例、series、decoder版本、frame映射、成功门和访问边界不变。

原IDEA/PREREG计划把pure-Python wheel路径直接加入`sys.path`。合成导入发现pydicom3.0.1会把包内`pydicom/data/urls.json`作为真实文件路径访问，zipimport报`NotADirectoryError`。这发生在任何ReMIND DICOM SEG下载前，不是数据结果。

修订为：把同一SHA256=`db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41`的官方wheel机械解压到`/tmp/aflow-remind-dicomseg-deps/pydicom-3.0.1`，不执行setup/pip install、不修改venv。解压后290 files/8,320,198 bytes的排序`path/bytes/sha256`紧凑JSON manifest SHA256必须为`49e2b56e2fac41da819f22e8801eb5b7afa5437f97f5eb48ddf3af43574c9946`；每次运行先重算。导入版本必须exact `3.0.1`且module path位于该目录。

该修订只改变包加载方式，不允许安装额外pixel decoder或放宽Transfer Syntax/SEG mapping门。若pydicom内置numpy handler不能解码冻结SEG，当前实现失败并停止。
