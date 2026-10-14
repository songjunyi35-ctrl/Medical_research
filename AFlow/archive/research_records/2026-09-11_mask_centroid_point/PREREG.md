# 预测掩膜几何中心点预注册

固定subject_balanced两seed epoch168、dev180/78、全部8反射；区域输出严格复用Flip8 mean logit>0。Point仅改为预测前景坐标质心的最近前景体素，欧氏距离平分取flat index最小；空mask保留原argmax。两seed Dice/区域计数必须exact，Point>=identity且分别>=169/162；失败不搜规则，不访问test。