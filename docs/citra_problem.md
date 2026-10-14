# CITRA 查询问题（Milestone 1）

目标论文暂名 *Certified Temporal Evidence Aggregation for Text-Augmented Time Series Forecasting*。
CITRA 是待验证的研究原型，当前没有真实预测收益、系统加速或会议录用级创新性的证据。

逻辑查询为 `q=(X_history,target,horizon,cutoff,epsilon)`。第一里程碑的执行接口接收已经由历史信息产生的
`q_z[p]`、UTC cutoff 和有限非负 epsilon。target/horizon 的条件化编码由后续预测模型实现，当前不训练编码器。
embedding 与 impact vector 都必须是固定的有限实数向量；不能从在线查询的未来真值计算影响向量。

合法文档必须有可靠的 available_at 及其来源证明，在 cutoff 前可访问，并采用当时可访问的版本。
event_start/event_end 描述被讨论事件的区间；预测未来事件的文本可以提前发布，它的事件时间不决定可见性。
未知 available_at 即使带有事件日期也排除。模拟数据显式标为 simulated，并必须启用 allow_simulated。

完整参考值为 `A_full = sum_visible exp(-||q_z-z_d||²/(2 sigma²))*v_d`，是带符号向量的完整求和。
语义 Top-K 不等价于这一参考值。近似执行返回 `A_approx`、聚合 L2 误差界、预测 L2 误差界和执行统计。

预测形式为 `Y=F_TS(X_history)+G(X_history,A)`。固定同一历史输入和同一组参数，如果 G 对 A 的
Lipschitz 上界为 L，则输出认证预算是 `L*aggregate_bound <= epsilon`。
调用者传入的 L 必须有效；默认 L=1 只认证聚合本身，不能自动认证任意预测网络。
当前 FixedLinearForecaster 用固定数值线性骨干和范数受控的线性文本头验证多变量多步输出；不是训练完成的预测模型。

epsilon=0 使用完整 oracle 精确遍历，空库返回零向量与零界。该证书只比较同一模型的近似输出和
完整合法文本输出，不是相对于未来真实标签的 MSE/MAE 保证。浮点认证的范围见 citra_theory.md。
