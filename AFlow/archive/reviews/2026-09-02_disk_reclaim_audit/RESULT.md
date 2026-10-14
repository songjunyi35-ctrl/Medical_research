# Train cache冗余与磁盘释放审计

日期：2026-09-02 UTC

## 结论

没有删除任何文件。已证明一个7.93GB历史train-only cache是当前972例主cache的逐病例byte-exact子集；若用户明确授权，未来可删除**精确目录**`workspace/BraTS/cache/t1ce_wt_n4_braincrop_resolution_20260818/formal_128/`，预计释放7,927,508,320 bytes，同时保留972例superset、raw、builder、checkpoint和全部报告。

这足以解除未来percentile-clipping单因素实验的磁盘阻塞：当前约12GB可用，授权删除后约20GB；新建约7.4GB train cache后仍约12GB余量。未授权前不得删除。

## 为什么它可被视为冗余

- Superset：`t1ce_wt_n4_braincrop_128_20260817/train`，972例，index SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`。
- Subset：`t1ce_wt_n4_braincrop_resolution_20260818/formal_128`，756例，index SHA-256 `d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d`。
- 两者都是canonical RAS、N4 shrink2、image-only nonzero brain crop、源非零总体z-score、128³ trilinear image与nearest-exact label；不使用GT定位。
- 756个case ID全部包含在972例cache中，但顺序不同，因此审计按case ID映射而不是按slot硬比较。
- 逐例结果：image byte-exact `756/756`，label byte-exact `756/756`，最大image绝对差`0.0`；共比较7,927,234,560 tensor bytes。
- `images.f32` 6,341,787,648 bytes、`labels.u8` 1,585,446,912 bytes、`index.json` 272,317 bytes均被审计；另有1,443-byte `RAW_REPRODUCTION.json`。整个目录约7,927,508,320 bytes。
- Superset不是压缩摘要，而是完整972例同口径tensor，因此删除subset不会丢失这756例模型输入；subset也可由raw与归档builder重建。

## 工具失败与修复

首次合成测试因安全守卫写成路径包含任意`test`就拒绝，pytest临时目录名自身含`test`而误报；真实路径不含该词，V1真实756例比较仍完成且全exact。随后在不改变比较逻辑的情况下，把守卫收窄为明确角色token `test270/sealed_test/confirmation`，合成测试1项通过，并从头生成V2真实报告。

V2机器报告`REDUNDANT_FORMAL128_AUDIT_V2.json` SHA-256 `015b44c5106d1918a8deb15992fac0ae38e473c657c84a7fcfa575250a86dce2`；runner SHA-256 `bf23506219ff4ee424d606d9556c171c105e30747f3a8df0af83e97c1ce0a04c`。

## 边界

- 只读取两个train-role cache；没有访问test270、confirmation、external cache或模型。
- 没有运行推理、训练或新预处理。
- 没有审计/建议删除checkpoint、report、raw NIfTI、UCSD external证据或其他未证明重复的cache。
- 删除必须由用户另行明确授权；届时仍应先复核目录、V2报告hash与`df -h`，只删上述单一目录并报告可恢复性。

## 其他大文件去重扫描

对`workspace/BraTS`下全部118个≥50MB文件计算完整SHA-256，合计读取62,853,842,170 bytes；整文件重复组为0，没有发现可仅凭相同hash直接去重的checkpoint或tensor。清单`LARGE_FILE_HASHES.tsv` SHA-256 `e39610c71fb2d1b0eb7343536960d8950f60e844ebbb127a2772c5141f96782c`。

这与7.93GB候选不矛盾：756例cache与972例cache长度不同、case顺序也不同，整体文件hash必然不同；前者的冗余证据来自逐case重排后的756/756 tensor byte equality。除这一已证明子集外，不建议删除其他大文件。
