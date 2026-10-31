# 字段字典

market_raw：独立真实行情表；每date/tic唯一。close必须与成交/持仓数量的公司行动口径一致。缺open不补造。
features_mgt：冻结预测特征通道；必须独立认证目标日、available_at、ticker映射与尺度。不能反向恢复/重新拟合scaler。
date：来源交易日期；decision_date为前一实际行情日收盘后决策日期。
nav/cash：成交后净值及现金；pre_nav为旧数量按成交时真实价格先标记的交易前净值。
shares：实际持有数量；target_weights为期望权重，actual_weights为费用/约束后实现权重。
buy/sell：实际买/卖成交额；commission=c*(B+S)，slippage=s*(B+S)，tax=tax(date)*S；cost三项相加。
gross_turnover=(B+S)/pre_nav，含初次建仓；无half-turnover混名。
unfilled_buy_value：资金不足/离散手数后的买入成交额差；blocked_symbols为显式mask禁止交易数，不以缺volume推断。
daily_return：nav_t/nav_(t-1)-1，含费用；第一行仅初始资金，收益NA不计入n。
AR/SR/Vol/MDD/Calmar：公式见backtest.py与protocol_changes.md。零分母NA。
status：completed为实际执行；blocked为有效性缺项；failed为实际尝试失败。synthetic_interface_only绝非论文结果。
research_approved=False：资格mask未通过复权/交易来源审查；eligible_by_known_listing仅已知上市检查，不代表整体合格。
