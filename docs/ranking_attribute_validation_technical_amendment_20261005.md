# 尺度核验技术修订

首次分析已完成全部排名和属性参考统计，在尺度输入检查处停止。原Objects365资格规则（scripts/audit_objects365_confirmation_source.py）允许边界超过图像最多0.001像素；本轮却对归一化坐标使用严格[0,1]断言。D中4图边界超过最多0.000891像素，属于原资格范围，没有新增图像或状态选择。

保留首次失败progress为initial_failure.json、原分析代码及input_freeze.sha256.json。新增独立恢复脚本，沿用D原资格的像素容差，其余池仍严格[0,1]。不改数据、模型、标签、排名指标、重采样种子或次数。报告原坐标越界图及像素幅度；同时检查原面积与裁到边界后的面积是否改变尺度枚举。若超出原容差或任何尺度枚举变动则停止。

恢复脚本与本修订单独冻结于technical_amendment.sha256.json。运行scripts/resume_ranking_attribute_validation.py复算同一统计，属于输入资格核验实现修正，不构成科学协议更改。
