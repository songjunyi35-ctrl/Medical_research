# Experiment idea: MOTUM-HGG29共享离散方向桥

Status: planned; input-only, no cache/model/performance.

## 1. 现在的问题是什么？

MOTUM-HGG29原性能协议在首例模型前停止：图像/双mask native shape相同、affine最大误差仅`2.384e-7`，29/29三文件离散orientation arrays也相同；但三个文件各自计算closest-canonical affine时，索引191/255的翻转把微小方向余弦差放大，2/29超过冻结`1e-5`门，最大`1.4488e-5`。0 cache/模型/性能。

## 2. 一句话直觉

三张对齐透明胶片要翻面时，用同一个翻面动作同时翻；不要让三台计算器按略不同的小数各自重算纸张原点。

## 3. 为什么选择这个idea？

- 29/29离散轴排列/翻转完全相同，说明这是header浮点传播，不是需要插值的真实方向冲突。
- Native最大误差`2.384e-7`远低于新冻结的`5e-7`严格门；共享离散变换不会移动或插值任何voxel。
- 旧失败只否定“每个文件独立生成canonical affine后再以`1e-5`比较”，没有测试从图像生成一次离散orientation并同步作用于mask。
- nibabel官方orientations API把`io_orientation`、`ornt_transform`、`apply_orientation`和`inv_ornt_aff`作为轴置换/翻转接口：[官方文档](https://nipy.org/nibabel/reference/nibabel.orientations.html)。

## 4. 具体怎么做？

每例读取native T1ce与两个mask；先要求shape exact、image-mask affine max≤`5e-7`，再要求三个`io_orientation`数组逐项exact。从图像orientation计算一次到RAS的离散transform，把同一个transform用`apply_orientation`施加到三个数组；shared RAS affine只由图像native affine和该transform计算。验证转换后的三个数组分别与其各自`as_closest_canonical`数组逐体素相同，但不再拿独立重算的mask affine作下游空间定义。

两个全新Python进程独立完成29例并保存数组SHA/shared-affine；finalizer要求两份完整JSON exact。全程没有N4、缩放、模型或性能。

## 5. 唯一变化与固定项

- 唯一变化：独立canonical affine生成→图像导出的共享离散orientation/shared affine。
- 固定：双官方source、HGG29顺序、原始arrays、mask文件语义与正类并集、native `5e-7`门、RAS目标、无插值、无N4/cache/model/performance。

## 6. 成功、失败和停止规则

- 成功：29/29 native shape exact、affine≤`5e-7`、orientation exact；共享转换arrays与各自canonical arrays exact；label域/WT非空；copy A/B所有数组和shared affine SHA exact。
- 失败：任一项失败即关闭；不改`5e-7`、不排病例、不重采样mask、不读取性能。
- 通过只授权另立新的MOTUM性能预注册；不自动复活旧实验。

## 7. 成本与风险

- 训练/GPU/model forward：0。
- CPU：两进程各读取29×3 NIfTI，预计秒级。
- 存储：JSON低于1 MiB，不保存转换数组。
- 风险：shared affine是明确的数据接口决定；即使arrays不变，也必须在未来结果中披露。极厚层grid caveat不由本桥解决。

## 8. 最终结果（完成后填写）

- 工程结果：29/29全部门过；copy A/B case payload exact；共享转换arrays与各自canonical arrays exact，0插值。
- 是否支持直觉：支持。失败来自独立canonical affine浮点传播，而非数组方向或真实网格冲突。
- 下一步：旧性能实验不复活；另立shared-orientation性能协议并重绑全部门。
