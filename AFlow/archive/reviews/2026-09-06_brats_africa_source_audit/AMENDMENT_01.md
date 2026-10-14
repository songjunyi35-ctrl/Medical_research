# Amendment 01: bounded Faspex directory retry

首次执行只列完根目录，第二层metadata请求因TLS EOF中断。此时没有inventory/report/datainfo归档，没有package payload、NIfTI、标签或模型访问。

唯一实现修改：每个Faspex目录分页请求发生`requests`连接异常或非成功HTTP时，最多尝试3次，前两次固定等待0.5秒/1.0秒；三次仍失败则终止。Package948、public-link、递归路径、分页token、文件正大小/唯一配对门和0 payload边界均不变。该重试只恢复metadata传输，不是病例/标签/结果重跑。
