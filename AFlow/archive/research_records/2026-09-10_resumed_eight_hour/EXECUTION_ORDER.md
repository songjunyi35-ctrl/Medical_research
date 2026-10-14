# 当前覆盖说明：本轮模型实验已经结束

旋转四模型与模糊两模型均完成168轮，正式开发及实际mask核验已归档，两种配方均未过联合门。全864旋转和固定seed11模糊测试均未准入，0新test。以下执行顺序保留为过程历史，不能再启动或重复任何本窗口训练/开发/测试入口。当前八小时窗口的剩余时间只用于证据与文档收尾；最终窗口是否结束以RESULT.json及其finished_utc为准，未生成前不能冒称已到八小时。

未来若获新工作窗口，先读当前交接与RESEARCH_SYNTHESIS.md，另注册新的目标编码/匹配对照研究；不要修改这次失败门或复用已经关闭的脚本当新实验。新fit子区标签cache已独立核验，但没有训练区域编码模型。

---

# 本窗口执行顺序（不是新的授权或延期）

截止2026-09-10 21:39:55 UTC，计算21:24:55前结束。已在运行的监督器不要重复启动；实际阶段查STATUS.json和各TRAINING_*.json。所有命令用项目venv、PYTHONDONTWRITEBYTECODE=1；CPU检查加OMP/MKL/OPENBLAS线程各1。默认终端失败时使用授权host /bin/sh，不能把环境故障误当科研已到上限。

## 当前主实验

scripts.run_small_rotation_training 已启动：seed11两臂完成并核SHA/导出，seed12两臂正在运行。四模型全部完成且监督器正常退出后才依次：

1. `python -m scripts.audit_small_rotation_training --final`
2. `python -m scripts.evaluate_resumed_with_masks --study small_rotation`
3. `python -m scripts.finalize_small_rotation`，随后`python -m scripts.verify_resumed_development_masks --study small_rotation`
4. `python -m scripts.publish_training_study_result --study small_rotation`
5. `python -m scripts.normalize_generated_archive_links --study small_rotation`
6. `python -m scripts.refresh_resumed_research_status`，`python -m scripts.resumed_research_snapshot --publish`，三个根规范validators。

不读first-seed单独dev，不改已经冻结的实现或参数。数据角色fit684/313与reused dev180/78。

## 主开发合法入选且预算充足

只能selected_arm合法、独立VERIFICATION通过后：smoke_small_rotation_full864 → freeze_small_rotation_full864 → run_small_rotation_full864。各模块均以`python -m scripts.<module>`运行。

full864完成后：audit_small_rotation_full864_training → freeze_small_rotation_test270 → evaluate_small_rotation_test270 → `verify_resumed_test270_panel --study small_rotation_full864` → `archive_resumed_test270_panel --study small_rotation_full864`。

全864训练会把原dev180消费为训练；之后不再跑备用fit开发。所有四test输出必须两榜归档，追加全864方法解释/当前状态。旧weak_elastic_full864分支仍关闭，不能运行旧入口。

## 主旋转无合法晋级资格时的唯一已注册备用

仅在旋转已结束且最晚19:00前、预算够时：smoke_image_blur → admit_image_blur → run_image_blur_training；两seed一起各168轮。

完成后：`audit_image_blur_training --final` → `evaluate_resumed_with_masks --study image_blur` → finalize_image_blur → `verify_resumed_development_masks --study image_blur` → `publish_training_study_result --study image_blur` → `normalize_generated_archive_links --study image_blur`。

模糊全门通过且本窗口尚无test面板才：freeze_image_blur_test270 → evaluate_image_blur_test270 → `verify_resumed_test270_panel --study image_blur` → `archive_resumed_test270_panel --study image_blur`。

固定test seed11，不择较好seed；对照是患者均衡旧模型，不是native。单窗口最多一个test面板。科学门与统计种子已写入TEST_PLAN，不得结果后修改。

## 已完成，不重跑

- 旧点研究限定复核：POINT_RECORD_CORRECTION.json/md；原坏收据保留，不能声称已独立重算点生成器。
- letterbox_fidelity：fit标签保形代价诊断完成，0模型、无训练准入；RESULT.md链接已修正并保留修正前副本。
- development_error_anatomy：四旧模型720开发前向完成，mask独立核验/解释完成；不加载新候选、不改训练参数；/tmp类别cache360MiB+mask180MiB保留。
- 全仓QUALITY_142506.json为当时921 passed，之后新增测试另有通过；最终要再跑run_resumed_quality_checks。

## 结束前

所有真实结果/未准入条件分支分开归档，补RESULT.md+机器JSON与方法卡，运行三个validators。若test运行，必须两榜更新且校验通过。最后创建本resumed窗口总RESULT，不重跑旧八小时finalize_data_processing_session_20260910（它会覆盖旧时间/报告）。不假称超过旧榜；新旧窗口及后补5760旧开发前向分开计数。

Goal工具仍blocked且create_goal拒绝unfinished goal；当前新工作来自用户明确新八小时授权，不把旧目标伪标complete来绕过。后续只在真实目标达成时更新complete。

新增工程观察器在正式开发访问前明确补充，原训练和数值评估代码仍hash-bound。两个原evaluate入口只由透明包装调用，不另行重复执行；额外保存180/90MiB实际mask且0额外forward。先原统计finalizer，再实际mask核验，再发布/晋级。
