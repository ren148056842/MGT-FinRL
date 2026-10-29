# v3 方法变化与会计范围

## 研究问题纠正

作者2026-10-25明确：MGT-derived observations are used during policy training in 2022. Validation and testing use observed market data for all policies. The comparison therefore evaluates the effect of forecast-augmented training on subsequent portfolio performance.

这段表述与v3实现一致。2023/2024不需要预测，不把原raw相等诊断当缺失预测。2023用于checkpoint选择，2024是此前已查看的固定历史评估期。冻结预测/图不重新训练或推理。

## 数据适配

train_features只在2022使用选定merge；2022之前各组统一raw预热。原银行merge2018—2021的替换值不进入v3预热。validation/test_features分别引用全raw历史生成的共享文件，包括2022年预热也全raw，防止预测残留到2023指标。market_raw始终独立，任何阶段的成交、净值与reward不使用预测价格。

维度保持(N+4)×N，字段为252日收益协方差与macd/rsi_30/cci_30/dx_30，不增加另一套预测表示、不纳入无历史可得性证据的估值/文本。不做全期fit或后向填充。

银行PAD派生统一为名称表PAB，CMB/CMBC分别保留；名称表的Wind来源不覆盖作者确认的每日Tushare来源。最新静态收盘价不用作历史行情。医疗映射从tech.csv读取。已知银行上市前伪历史从所有预热中排除，固定股票池仍有幸存者偏差，不能声称消除。

## 增强器时间边界

离线训练增强允许用到2022训练期信息，不能单因forecaster/scaler截止2022而将2024 RL结果判为泄漏。原银行生成代码保存输出显示close matrix为1215×15、来自bankclose.csv；同一预测数据预览覆盖2018—2022。helper虽有全base回退分支，保存输出不是走2024回退，因此不能把“可能回退”冒充已证明评估信息污染。部分图/预测器的真实生成截止仍未知，按未知披露，不称无泄漏。2022预测精度自己的样本外有效性与2024 RL策略评价分开讨论。无已证实2023/2024污染时不因此全局停训；若出现确凿污染证据，只暂停受影响比较。

选定 canonical finrl_csv 为原通用消融生成族，不按历史收益选best_policy目录。非正预测只记录，不裁剪、不替换为真值；只有派生状态确实非有限才停对应变体。所有原CSV/NPY/H5/notebook不变。

## 会计范围

数据来源与不复权口径来自author_confirmed，不冒充第三方认证。主结果命名unadjusted_price_cost_sensitivity，状态cost_analysis_scope_limited、economic_return_validated=false。允许真实RL训练和评估，实际完成数不受paper_eligible字段掩盖。

尚无完整现金分红、送转拆股及支付/新增股可交易日期；当前固定分数份额按不复权价格计价，不是total_return、完整经济收益或实盘约束模型。除权跳变不解释为真实投资损失。最初环境变量未发现Tushare凭据；训练启动后进一步检查发现SDK已有存储凭据并进行最小化事件补查。接口返回每小时1次限额，30只股票事件不完整，详见tushare_event_fetch_status.json和BLOCKED_ITEMS.md。不显示token、不索取或记录密钥；不把缺事件当零，缺项不阻止本轮范围已授权的价格费用实验。事件复核保持独立，不修改已运行的训练奖励。

无open，采用t收盘决策、t+1真实收盘执行：旧持仓承担此前价格变化。全现金100万，初次建仓收费，按漂移后实际买卖金额计算c/s及卖出税，滑点只扣现金，不免费末日清仓。无融资/卖空。分数份额、未知停牌/涨跌停/最低佣金/流动性/市场冲击等限制披露，缺volume不等同停牌。

## 训练预算与选择

算法/学习率/默认网络/gamma沿用原设置与现有SB32.4.0。奖励净单期收益×0.1，不声称仍为财富奖励，不做reward网格。20/40/60/80/100%检查点在2023raw F1净Sharpe选择，平分最早；测试表现不选择策略/seed/参数。PPO按2048 rollout向上取整；A2C按n_steps取整；TD3/DDPG默认按完整episode采样时也可能超过指定步数不足一个episode。每个checkpoint与最终真实num_timesteps均记录，不把目标预算当实际步数。

优先18训练及7档费用先保存并汇总，再复用checkpoint扩展主比较60，固定PPO消融新增36，总RL上限96，不重复为费用训练。一次真实A2C短跑单列，不计主训练。

## 非RL与统计

等权基准各行业只做一份价格路径；不要求测试预测Top3。新增Ridge分别用2022 Vanilla/A0状态拟合，alpha=1、k=3固定。StandardScaler只fit训练输入，标签raw close[t+2]/close[t+1]-1且完全落在2022；验证/测试输入全raw，同一延迟成交。

指标252年化、日收益rf0及2%假设Sharpe敏感性、ddof1、包含初始资本的正MDD、零分母Calmar为NA。AR/MDD费用变化用百分点，SR用绝对差，分别相对F0/F1，不强迫指标单调。

配对bootstrap同市场日期索引作用于两策略及各seed，逐seed差后平均；主block10/B2000，5/20只作廉价敏感性。区间条件于固定merge/策略/股票池/时期，不把seed×day当独立样本，不虚构p值。固定merge多RL种子不等于重训完整预测流水线；半年度切片不是walk-forward。

R1.2费用计算完成后仍须附会计限制，不能全面fully_resolved；R1.3/R2.6独立验证+固定PPO；R1.4/R2.5配对随机种子+区间部分回应；R5.4 Ridge同输入对照。初始化、跨股票注意力、无WM、图定义/来源、完整预测随机性等仍未开展/未解决；R1.1作者已处理。
