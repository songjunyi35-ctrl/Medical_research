# 精确占比训练标签：缓存工程完成

fit684扫描/313人全部完成，原生label SHA与旧记录一致，原nearest-exact硬WT逐例byte exact。生成的占比文件已逐例重读哈希与统计，范围[0,1]、非空总量、坐标/顺序和量化门全部通过。原影像、硬标签与评分协议没有修改。

float64最大相对质量误差3.36e-16；float16最大4.02e-05、平均1.93e-06。旧硬标签在同一输出格体积换算下平均质量误差0.008549，只是数据表示诊断，不是模型Dice。

部分占比格总数12000530；占比>.5后整WT为空0例。该硬化结果与旧hard的平均掩膜重叠0.973320，不用于改阈值/核。旧公共crop外仍有4个WT体素，原样保留记录、不排例；质量守恒指裁剪内。

成本：builder 2356.10s、peak RSS 1.094GiB；标签文件2868903936bytes=2.672GiB，位于临时目录，0GPU/model/dev/test。原先数分钟估计偏乐观，实际成本如实记录。

这只证明构造正确且保留几何占比信息，不是软标签提升分割的证据，也不是医师信心或概率校准。它是原标注单位格恒定占据的数学模型，并非完整MRI扫描点扩散函数模型。图像原三线性重采样仍固定，不能声称整条成像模型已纠正。

下一步仍为条件性训练：弱形变不能晋级且时间充足才执行两个固定seed；否则作为已准备的备用数据表示保留。部署不需要此标签cache；/tmp不保证跨环境长期保留，已归档builder/原生label SHA/index/receipt以便重建。

[七问](IDEA.md)、[预注册](PREREG.md)、[cache前冻结](CACHE_PREACCESS.json)、[cache收据](CACHE_RECEIPT.json)、[index](CACHE_INDEX.json)、[验证](CACHE_VERIFICATION.json)。

Target SHA `32d7815fbfc79e5bfd42e9872808b6afb439e324151068a812c79a0b49d51ee6`；cache receipt SHA `dbc3fbef0c8ff2be3fdcf5b3ab1d77a5a8dec2435e2228d6dc7bc70d52427064`。
