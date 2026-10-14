# 固定输出与独立核验

四组二值mask共270MiB，C顺序、little位序；冻结数据/checkpoint/脚本/统计门/SHA后仅一次正式读取。实际mask独立重算交并/体积/Dice，保存坐标核GT命中及患者簇区间。不保存原始logits，不能声称独立重算argmax。

先verify_resumed_test270_panel --study image_blur，再archive_resumed_test270_panel --study image_blur，保留开发RESULT原样，测试另归档到2026-09-10_image_blur_test270。所有四输出都入榜；不能用负结果不更新或用正点估计代替统计证据。当前仅准备，尚未访问test。
