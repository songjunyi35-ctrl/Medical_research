# 备用模糊：开发预测mask归档补充

正式模糊训练和开发均尚未执行。仅补充保存两个固定seed候选的原logit>0掩膜（90MiB），原图像模糊参数、标签不变规则、epoch168、阈值/Point、统计种子和全部晋级门不改。

透明观察器返回原输出对象，先合成和旧fit8 GPU精确输出检查；两模型完成后DEV_MASK_PREACCESS额外绑定实际包装入口、冻结原入口与checkpoint，调用一次evaluate_image_blur.main，0额外forward。原raw不回写，随后独立mask核对360例计数/Point。未保存logits，不声称重算argmax。

只用已有哈希固定的类别cache解释错误，不据此调参。失败保留访问计数，不重复开发补mask、不按部分成绩选择。实际命令`python -m scripts.evaluate_resumed_with_masks --study image_blur`，原统计finalizer之后运行verify_resumed_development_masks。旋转主实验优先条件与单窗口最多一个test面板保持原样。
