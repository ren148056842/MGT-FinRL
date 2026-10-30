# v3 复现与恢复说明

正式入口为 `revision_backtest/run_revision_v3.py`；`run_revision.py` 保留 v2 行为，不用于 v3。解释器固定为 `C:\Users\perfe\.conda\envs\finrl-meta\python.exe`。所有模型严格顺序运行，单模型一个 CPU 线程；这是作者在本次运行中的明确选择。不得同时启动第二个训练入口。

全流程（仅在没有现行训练进程时执行）：

```powershell
& 'C:\Users\perfe\Desktop\2\xiu\revision_backtest\run_all_v3.ps1'
if ($LASTEXITCODE -ne 0) { throw 'v3 pipeline failed' }
```

按阶段、算法、行业、种子恢复示例：

```powershell
$revisionPython = 'C:\Users\perfe\.conda\envs\finrl-meta\python.exe'
$revisionEntry = 'C:\Users\perfe\Desktop\2\xiu\revision_backtest\run_revision_v3.py'
& $revisionPython $revisionEntry train --phase 1 --sector healthcare --algorithm sac --seed 1
if ($LASTEXITCODE -ne 0) { throw 'training command failed' }
& $revisionPython $revisionEntry evaluate --phase 1
if ($LASTEXITCODE -ne 0) { throw 'evaluation command failed' }
& $revisionPython $revisionEntry report
if ($LASTEXITCODE -ne 0) { throw 'report command failed' }
```

成功任务由 receipt 的协议、输入、代码、环境及产物哈希共同保护。完成模型直接复用；完成费用档位直接复用。失败任务不自动重复训练。尚在进行中的 SAC/TD3/DDPG 不可因临时检查而结束进程：当前检查点不含可完整恢复学习的 replay buffer，不能将重启冒充无损继续。checkpoint 用于推理与验证选择；不是本轮授权的重复训练入口。

`all` 在 18 次重点训练后生成 `results_priority_v3.zip` 和 `priority_completion_counts.json`，然后才进入其余主比较与固定 PPO 消融。总主训练上限 96；短跑单列，不计为正式训练。训练阶段的 5 个 checkpoint 是同一模型的一次训练过程，不是 5 次独立训练。

## 产物索引

- `models/<task>/training.log`、`validation.csv`、`checkpoint_metadata.json`：真实学习、验证选择、实际步数与超参数；权重留在本地，不放入交付 ZIP。
- `results/evaluations/<task>_<fee>/`：逐日轨迹、持仓、成交、指标和成功收据。
- `results/nonrl/`：基准与 Ridge；`execution_records.jsonl` 按真实类型分开计数。
- `results/cost_sensitivity.csv`：同一 checkpoint 的逐种子七档结果及相对 F0/F1 的差值；`paired_bootstrap.csv` 条件于固定增强输入，无完整预测器重训含义。
- `analysis_tools/verify_completed_artifacts.py`：完成产物的只读会计、指标、日期、模型选择与哈希核验；运行结果在 `verification/`。
- `analysis_tools/event_accounting_review.py`：公司行动来源齐全时才执行的固定策略复核。与正式训练分开，不改奖励，不重训，不把未纳入事件的学习升级成经济收益训练。
- `regression_tests.log`：必要回归；`event_accounting_review/event_ledger_tests.txt`：单独的合成事件账本测试，均不能计为正式训练。

## 逐日字段和范围

`date` 为真实成交/估值日，`decision_date` 为上一真实交易日收盘决策日。首行仅保存初始现金，不是一个投资收益期，所以 `daily_return` 和 `reward` 留空；统计从第二行开始。`nav` 是费用后净值，`pre_nav` 是当日调仓前净值。`buy/sell` 为实际成交额；`commission/slippage/tax` 分项只扣一次。`gross_turnover=(buy+sell)/pre_nav`，不是单边换手。`quantity` 是不复权价格口径的固定分数单位。最后一日按市值保留持仓，不自动免费平仓。

本轮主要结果均为 `unadjusted_price_cost_sensitivity`。未纳入公司行动时，原始除权价格变化不是完整股东经济损益。不能称 total return；不能与另一事件口径混表。AR 为 252 日复利年化，主 Sharpe rf=0、样本标准差 ddof=1；rf=2% 用同一轨迹另算。MDD 是包含初始资金的正损失比例。AR/MDD 的差值列以百分点表示，SR 的差值为绝对差。

`paper_eligible=false` 不等于未执行。以 `actual_real_data_train/eval` 的完成收据统计实做次数，另列 `economic_return_validated=false`。尚未跑到的任务是 pending，不是数据 blocked。
