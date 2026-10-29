# v3 实际执行摘要

{
  "actual_real_data_train": 96,
  "actual_real_data_eval": 492,
  "real_api_short_run": 1,
  "nonrl_fit": 4,
  "nonrl_eval": 56,
  "regression_tests": 1,
  "failed": 0,
  "blocked": 0,
  "economic_return_validated": false,
  "accounting_scope": "unadjusted_price_cost_sensitivity"
}

MGT训练观测仅2022增强；2023验证与2024测试及其预热历史对所有策略完全相同且全真实。真实账本在训练/验证/测试始终使用raw。

Tushare不复权来自author_confirmed；公司行动/现金分红尚未纳入。结果是unadjusted_price_cost_sensitivity，cost_analysis_scope_limited，非total_return，不声称完整经济收益或全面实盘约束。2024为已见固定历史评估期。

## 重点比较F1实测（各seed均保留）
- bank a2c A0：n=3；AR均值 0.3360；SR 1.6361；MDD 0.1067；相对F0 delta_AR -1.3865个百分点、delta_SR -0.0549、delta_MDD 0.0412个百分点。
- bank a2c Vanilla：n=3；AR均值 0.3479；SR 1.6822；MDD 0.1048；相对F0 delta_AR -1.4819个百分点、delta_SR -0.0581、delta_MDD 0.1235个百分点。
- healthcare ppo A0：n=3；AR均值 -0.2032；SR -0.3603；MDD 0.3682；相对F0 delta_AR -1.8840个百分点、delta_SR -0.0579、delta_MDD 0.6888个百分点。
- healthcare ppo Vanilla：n=3；AR均值 -0.2098；SR -0.3737；MDD 0.3765；相对F0 delta_AR -1.4792个百分点、delta_SR -0.0457、delta_MDD 0.5498个百分点。
- healthcare sac A0：n=3；AR均值 -0.1945；SR -0.3226；MDD 0.3616；相对F0 delta_AR -1.2468个百分点、delta_SR -0.0377、delta_MDD 0.4970个百分点。
- healthcare sac Vanilla：n=3；AR均值 -0.1892；SR -0.3076；MDD 0.3622；相对F0 delta_AR -0.7417个百分点、delta_SR -0.0223、delta_MDD 0.3387个百分点。

## 解释边界
不同表示消融是训练增强输入的比较，不是测试时实时图删除干预；三RL种子条件于固定merge，不是完整预测流水线重训。配对bootstrap共用日期块，逐seed差再平均，无p值。非正预测不裁剪，不用于账本；具体非有限状态只暂停对应变体。

R1.2已有计算也须附上述会计限制，不能把全部现实约束勾为fully_resolved。未知预测器/图来源不足不被误称为已证实评估污染；也不宣称无泄漏。R1.1作者已处理。

论文对应实现表述：MGT-derived observations are used during policy training in 2022. Validation and testing use observed market data for all policies. The comparison therefore evaluates the effect of forecast-augmented training on subsequent portfolio performance.
