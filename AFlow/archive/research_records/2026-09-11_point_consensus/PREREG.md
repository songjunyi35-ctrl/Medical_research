# 多视图点共识预注册

固定subject_balanced两seed epoch168、同一dev180/78、完整八镜像。区域mask使用原Flip8平均logit；Point使用八候选argmax的支持票数→平均logit tie-break。无GT参与。两seed均须Point>=identity且分别>=169/162，Dice保持原Flip8；失败关闭，不调邻域/票权。测试集仍禁止访问。