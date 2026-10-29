# Guided输入阶段交接

- 原80训练完全不动。两输入beta .25/.5，k5 eps .04，所有CPU/160-168/batch8/尾4和纯合成尾6门通过后冻结预算；恰1728新增病例前向，170.83秒；4固定decoder完成，18:36:24释放GPU1。无270。
- best_new guided_beta025_covariance Scan0.769239635、Subject0.758959997、Point99/108。对原cov −0.000595876，患者95%区间[-0.001013249,-0.000187495]；所有4项低于原cov，五折无折选guided，不推荐因新而升选。此前原研究对预处理的负证据一并保留。
- E/DEPLOYMENT_RECIPE.candidate完整等于RESULT.best_new.recipe；稳定id用于RAW.per_configuration。E/INPUT_DEPLOYMENT_RECIPE.input_recipe为单一beta025 spec；source_sha256绑定core。部署仅选一个w32输入的8视图，RACE base8完全不变，2模型/16forward，不把两个w32输入加在一起。
- 最终model_id w32e171_guided、base_model_id w32e171、atom_source guided_input_w32；cp/架构/构造与原W相同，模型项input_recipe/path/SHA绑定规范文件。WREADY为B/final270/models/w32e171_guided/ATOMS_READY.json；只真实rootFINAL选择后直接生成选中滤波8×270=2160，不先做无用原W8。
- fusion固定075或原selected_covariance完整配方，cov的algorithm_recipe_path/SHA/sourceSHA仍绑定原cov文件；输入core归模型inputrecipe绑定。新final入口edge_input_b/final270.py支持全部legacy/cov/contra与新guided；A独立_healthnorm单模可共存但不混入fusion。完整270保存掩膜二次audit齐全后才出版RAW_FUSION。
- B/FINAL270_GUIDED_ENTRY_FREEZE为实际源码冻结，GUIDED_INPUT_ENTRY_FREEZE为同源SHA兼容引用，均保留。CPU新路由与4个缓存decoder精确score/mask/Point门通过。新final watcher16408仅替换空闲15136；收据含前后PID/cmdline，snapshot9260/cov11776/contra15135不动。
- final_snapshot fixed transfer16407排在contra完成后，仅换final-budget新RACE primary，伙伴始终w32e171_guided、配方单一已选，不重搜4网格；正best0写NO_NEW_MODEL退出不重造缓存。结果schema与前轮相同，RAW键candidate.id。
- GT事后oracle仅错误诊断、绝非模型成绩或可达0.8，不进入排行榜。所有历史冻结/权重/源码与临床载荷保留，无跨目的地转移。
