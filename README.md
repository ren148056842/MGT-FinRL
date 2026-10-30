# MGT-FinRL

Code, data, and results for the manuscript **"MGT-FinRL: A Multi-layer Graph Neural Network-Transformer Based Reinforcement Learning Investment Strategy"** (Zhiping Chen and Xiaoping Ren, Xi'an Jiaotong University; manuscript APIN-D-26-10249, *Applied Intelligence*).

> ⚠ This repository is provided for peer review purposes only.
> All rights reserved. No part of the code or data may be used, reproduced, or distributed without explicit permission.
> The data files are provided for verification of the reported results only and remain subject to the terms of their original providers (Tushare and Wind).

The release cited in the revised manuscript is tag **`v2.0-revision`**.

## Repository layout

| Path | Contents |
|---|---|
| `*.ipynb` (root) | Notebooks of the original submission: graph construction (`share_holding_graph`, `Q&A_graph`), forecasting models (`GNNtransformer`, `GCN-transformer`, `GAT-Transformer`, `GNNGRU`, `GCNGRU`, `GATGRU`), and the original RL experiments (`FinRL`, `PPO,A2C,DDPG，AD3,SAC`). |
| `banksector.zip`, `healthcare.zip`, `data.zip` | Archived raw inputs: prices, volumes, valuation ratios, top-ten shareholder lists, and investor-relations Q&A records. |
| `graphs.zip` | Figure files of the manuscript. |
| `revision/` | Everything added for the revision (described below). |

### `revision/`

| Path | Contents |
|---|---|
| `revision_backtest/` | Revised backtesting ledger, environment, training, and evaluation code (`run_revision_v3.py` is the entry point; `v3_*.py` and its `v2_*.py` dependencies; `backtest.py` holds the reference ledger, cost settings, metrics, and bootstrap). Tests: `test_v3.py`, `test_backtest.py`. |
| `revision_backtest/runs/v3_train_only/` | Protocol (`protocol_v3.json`), resolved hyperparameters, selected inputs with SHA-256 hashes, run status, execution records, environment (`environment.json`, `pip_freeze.txt`), verification outputs, and `prepared.tar.gz` (training-observation tensors and non-RL models). |
| `revision_backtest/runs/v3_train_only/results/` | Summary tables (`*.csv`, `*.tex`), `evaluations.tar.gz` (daily trajectory, trades, metrics, and receipt of all 492 policy evaluations: 96 trained policies × cost settings), and `nonrl.tar.gz` (equal-weight and Ridge baselines). Daily position files (about 1 GB) and policy checkpoints (about 2.4 GB) are not included; they are available from the corresponding author on request. |
| `bank sectordata/`, `haelthcaresectordata/` | Inputs used by the revised pipeline: Tushare daily panels (`*trade20241231.CSV`), relation matrices (`*.npy`), valuation-ratio tensors, and the archived 2022 forecaster outputs for variants A0–A9 (`ablation*/finrl_csv/`). Folder names are kept as the code expects them. |
| `notebooks/` | Forecasting and RL notebooks used in the revision, including the original eight-run MGT notebook, the revised (chronological, MSE) forecaster and ablation notebook, and the banking Kendall-graph notebook. |
| `forecaster_runs/` | `forecaster_runs.tar.gz`: per-seed predictions (scaled and, for healthcare, in prices), targets, training histories, and test errors of the revised forecaster for variants A0–A9 (seeds 1–3, both sectors). The forecaster outputs in `ablation*/finrl_csv/` are the average of the three seeds' predictions. |
| `analysis/` | Scripts that produce the revised tables and figures from the stored outputs (`boot_all.py`, `tables.py`, `fig_nav_small.py`, `fc_metrics_full.py`, `verify_inputs.py`) and their outputs. |
| `logs/` | Text dumps of the archived notebooks with their recorded outputs. |

## Revised evaluation protocol (summary)

- **Data.** Two fixed universes of 15 A-share stocks (banking; healthcare innovation), Tushare unadjusted daily prices; dividends are not credited.
- **Splits.** RL training on 2022 (242 trading days), checkpoint selection on 2023 (checkpoints at 20/40/60/80/100% of the budget, highest net Sharpe ratio), evaluation on 2024 (242 trading days).
- **Agents.** PPO, A2C, SAC, TD3, DDPG (Stable-Baselines3 2.4.0), each trained with vanilla and MGT-augmented 2022 observations, seeds 1–3 (60 runs); fixed-PPO ablation over variants A1, A2, A6–A9 (36 runs).
- **Execution and costs.** Decision after the close of day *t*, execution at the close of day *t*+1; all-cash start of 10^6; long-only softmax weights; cost = (commission + slippage) × traded notional + stamp duty on sales (10 bp before 2023-08-28, 5 bp after). Settings F0–F6 = 0/0 (no duty), 3/2, 5/2, 10/2, 20/2, 3/0, 3/5 bp.
- **Reward.** 0.1 × net daily return computed from observed prices; γ = 0.99.
- **Statistics.** Paired circular block bootstrap over 2024 days (block 10, 2,000 resamples, seed 1729).

## Notes on the forecasting module (see Section 6.3 of the revised manuscript)

- Each MGT model predicts one price series of the **same day** for all stocks from that day's PE-TTM, PB, PS, and PCF (input window of one day). The ratios are computed from that day's close, so the forecast horizon is zero. The archived code and outputs are those of the closing-price model; the RL experiments replace only the 2022 closing prices.
- The forecasting comparison of the original submission (eight runs, random split of trading days, cross-entropy training) could not be reproduced from the archived notebooks and is withdrawn.
- The 2022 outputs used in the RL experiments come from the revised implementation in `notebooks/GNNtransformerbankclosenew-8RUNS消融_FIXED.ipynb` (MSE loss; training 2018-01-02 to 2021-03-18, early stopping to 2021-12-31, test 2022). `analysis/fc_metrics_full.py` evaluates them against observed prices (manuscript Table 2).

## Known data issues (disclosed in the manuscript)

- The banking shareholder file covers 13 of 15 banks and labels China Merchants Bank as China Minsheng Bank; CMB, PSBC, and CZB are isolated in the banking shareholder graph.
- The high and low prices for 2022 in the archived banking forecaster files differ from the observed values on about a quarter of the stock-days.
- Shareholder lists are single snapshots published in 2023 (healthcare: as at 2022-12-31; banking: latest lists at export, June 2023).

## Reproducing the revised results

```bash
# Python 3.8; finrl 0.3.4, stable-baselines3 2.4.0, torch 2.0.1, gymnasium 1.0.0,
# numpy 1.23.1, pandas 1.4.3, stockstats 0.4.1, scikit-learn 1.1.1 (full list: pip_freeze.txt)
cd revision/revision_backtest/runs/v3_train_only/results && tar xzf evaluations.tar.gz && tar xzf nonrl.tar.gz && cd -
python revision/analysis/verify_inputs.py      # graph, Q&A, and shareholder checks
python revision/analysis/fc_metrics_full.py    # Table 2 (forecaster evaluation)
python revision/analysis/boot_all.py           # Table 4 (paired bootstrap)
python revision/analysis/fig_nav_small.py      # Fig. 5
# Full retraining (about 16 CPU hours, one thread per run):
python revision/revision_backtest/run_revision_v3.py all
```

The training entry point records SHA-256 hashes of inputs, code, and environment in each run's receipt and reuses completed runs.

## Contact

Zhiping Chen (corresponding author), zchen@mail.xjtu.edu.cn
