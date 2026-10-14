# CUDA/UVM故障：正常与异常worker的只读对照

现有证据强烈指向特定节点的平台基础环境，不能据此称整个平台故障。

| 项目 | 正常1104883 | 故障1105011 |
|---|---|---|
| 同一项目Python环境 | PyTorch2.7.1+cu126 | PyTorch2.7.1+cu126 |
| NVIDIA驱动版本 | 535.161.08 | 535.161.08 |
| 宿主内核 | 5.4.143.bsk.5-oci-amd64 | 5.15.152.bsk.10-sign-amd64 |
| nvidia_uvm模块 | 已加载 | 未加载 |
| /dev/nvidia-uvm | 存在 | 不存在 |
| 驱动cuInit | 0（成功） | 999（失败） |
| PyTorch CUDA | 可用，训练正常 | 不可用，张量创建失败 |

模型代码在裸驱动cuInit检查中并未执行，因此可以排除训练逻辑造成该初始化失败。相同Python环境与驱动版本在正常节点可用，也不支持直接重装PyTorch或更改模型来修复。两边都会显示driver/toolkit兼容模式提示，该提示本身不是本次根因的证明。

UVM是NVIDIA统一内存相关的内核驱动组件，/dev/nvidia-uvm是其设备接口，不是项目Python包。故障节点不只是缺设备文件，模块列表也没有加载UVM；更倾向宿主机驱动初始化环节，但具体是模块安装/构建、自动加载、加载失败，还是平台运行时配置，仍需宿主机日志确认。内核版本不同是线索，不足以断言5.15不受支持。999是通用初始化错误，不能只凭错误码认定UVM根因。

nvidia-smi基于NVML管理接口，能查询显卡不等于CUDA计算已可用；需要实际CUDA初始化/张量或内核验证。官方依据：[NVML说明](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html)、[NVIDIA模块与设备创建工具](https://github.com/NVIDIA/nvidia-modprobe/blob/main/nvidia-modprobe.1.m4)。

几个失败worker端点IP相同，可能被反复调度到同一故障宿主机；具体物理归属由平台确认。建议平台在宿主机检查nvidia_uvm模块是否匹配当前内核、加载错误及内核日志，并验证容器设备注入；或排除故障宿主机后改派。普通用户容器中无法代替宿主机驱动修复；没有实施驱动卸载、重装、模块重置或GPU重置。

本轮没有新病例像素、模型患者推理或test/outer/confirmation访问，既有训练继续。[机器对照](RESULT.json)。
