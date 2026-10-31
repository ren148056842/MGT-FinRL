# MGT-FinRL

Code, data, and results for the manuscript **"MGT-FinRL: A Multi-layer Graph Neural Network-Transformer Based Reinforcement Learning Investment Strategy"** (Zhiping Chen and Xiaoping Ren, Xi'an Jiaotong University; manuscript APIN-D-26-10249, *Applied Intelligence*).

> ⚠ This repository is provided for peer review purposes only.
> All rights reserved. No part of the code or data may be used, reproduced, or distributed without explicit permission.
> The data files are provided for verification of the reported results only and remain subject to the terms of their original providers (Tushare and Wind).

The revised manuscript refers to the branch **`revision-v3.0`** of this repository.

## Repository layout

| Path | Contents |
|---|---|
| `*.ipynb` (root) | Notebooks of the original submission: graph construction (`share_holding_graph`, `Q&A_graph`), forecasting models (`GNNtransformer`, `GCN-transformer`, `GAT-Transformer`, `GNNGRU`, `GCNGRU`, `GATGRU`), and the original RL experiments (`FinRL`, `PPO,A2C,DDPG，AD3,SAC`). |
| `banksector.zip`, `healthcare.zip`, `data.zip` | Archived raw inputs: prices, trading values, valuation ratios, top-ten shareholder lists, and investor-relations Q&A records. |
| `graphs.zip` | Figure files of the manuscript. |
| `revision/` | Everything added for the revision (described below). |

### `revision/`

| Path | Contents |
|---|---|
| `revision_v4/forecaster/` | **Re-implemented forecaster** used in the revised manuscript: `fc_data.py` (inputs, targets, masks, walk-forward folds, standardisation), `fc_models.py` (MGT, ablations A1–A10, initialisation variants I1–I3, and the five graph baselines), `fc_run.py` (training, ensembles, and the 2022 merges used for RL training). |
| `revision_v4/forecasts/` | All forecasting outputs: `runs/<sector>/<fold>/<model>/seed<k>/` (test and validation predictions `pred.npz`, `history.csv`, `meta.json` with timings and parameter counts, `weights.h5`), ridge and previous-day forecasts, seed ensembles (`ensembles/`), and the 2022 merges (`merges/*.json` receipts and `merges_csv.tar.gz`). |
| `revision_v4/*.ps1`, `*.cmd`, `logs/` | Scripts that ran the experiments on Windows and their logs. |
| `revision_backtest/` | Backtesting ledger, environment, RL training, and evaluation code. Entry points: `run_revision_v4.py` (current experiments; `v4_*.py`, reusing `v3_data.py`, `v3_ledger.py`, `v2_*.py`, and `backtest.py`) and `run_revision_v3.py` (earlier experiments, whose vanilla runs are reused). Tests: `test_backtest.py`, `test_v3.py`, `test_v4.py`. |
| `revision_backtest/runs/v4_forecast_train_only/` | RL experiments of the revised manuscript: protocol (`protocol_v4.json`, with SHA-256 hashes of all code files), input manifest, execution records, per-run training logs and validation results (`models/`), training observations (`prepared.tar.gz`), and results (`results/evaluations.tar.gz`, `results/nonrl.tar.gz`, summary CSVs). |
| `revision_backtest/runs/v4_alpha_init/` | Fixed-PPO runs with the initialisation variants I1–I3 of the relation scalars. |
| `revision_backtest/runs/v3_train_only/` | Earlier execution of the same RL code; its vanilla runs are reused in the revised manuscript (the vanilla training inputs are identical, which `test_v4.py` checks element by element). |
| `analysis_v4/` | Scripts that produce every table and figure of the revised manuscript from the stored outputs (`run_analysis.sh`), and their outputs (`results/`). |
| `bank sectordata/`, `haelthcaresectordata/` | Inputs: Tushare daily panels (`*trade20241231.CSV`; the column `volume` is the daily trading value in CNY), relation matrices (`*.npy`), and the archived forecaster outputs of an earlier implementation (`ablation*/finrl_csv/`, not used in the revised manuscript). Folder names are kept as the code expects them. |
| `notebooks/`, `forecaster_runs/`, `analysis/`, `logs/` | Materials of the earlier revision stage: archived notebooks, the outputs of an earlier forecaster implementation, and their analysis. |

## Revised design (summary)

**Forecaster.** At the close of day *t*, MGT forecasts log(H/C<sub>t</sub>), log(L/C<sub>t</sub>), and log(C/C<sub>t</sub>) for days *t*+1 and *t*+2 from a 20-day window of eight features per stock (log return, log high/close, log low/close, log trading value relative to its 20-day mean, and the yields 1/PE, 1/PB, 1/PS, 1/PCF). The three relation graphs are used as constructed. The relation scalars α<sub>r</sub> are initialised at 1/ρ(A<sub>r</sub>). Training uses masked MSE on standardised targets, Adam (1e-3), batch 32, ≤100 epochs, and early stopping (patience 10), with seeds 1–3; the downstream forecasts are seed averages. The evaluation uses walk-forward folds with test years 2022, 2023, and 2024 (training from 2018 to two years before the test year, early stopping on the following year). The benchmarks are the previous day's prices and a pooled ridge regression.

**RL.** PPO, A2C, SAC, TD3, and DDPG (Stable-Baselines3 2.4.0), seeds 1–3, are trained on 2022, checkpointed on 2023 (net Sharpe ratio), and evaluated on 2024. The MGT-enhanced agents are trained on 2022 observations in which each day's H/L/C are replaced by the one-step-ahead forecasts of the 2022 fold. Validation and test states are observed prices for all agents. Execution is at the next close, with commission, slippage, and stamp duty (settings F0–F6), and the reward is 0.1 × the net daily return. A fixed-PPO ablation covers A1, A2, A6–A9, and I1–I3. The non-RL rules are equal weighting, ridge top-3 rules on the training observations, and top-3 rules on the forecasts of each of the 17 forecasters.

## Reproducing the results

```bash
# Python 3.8 environments: forecaster with TensorFlow 2.11; RL with stable-baselines3 2.4.0, torch 2.0.1,
# gymnasium 1.0.0, numpy 1.23.1, pandas 1.4.3, stockstats 0.4.1, scikit-learn 1.1.1 (pip_freeze.txt)
sh revision/analysis_v4/run_analysis.sh          # all tables and figures from the stored outputs
python revision/analysis/verify_inputs.py        # graph, Q&A, and shareholder checks
# Full re-run (Windows paths are set in the scripts; about 4.5 h of forecasting and 8 h of RL on a laptop CPU):
#   revision/revision_v4/run_all_v4.ps1   (forecasts, then RL in parallel with the remaining forecasts)
```

Each RL run records the SHA-256 hashes of its inputs, code, and environment in a receipt, and completed runs are reused.

## Known data issues (disclosed in the manuscript)

- The banking shareholder file covers 13 of the 15 banks and labels China Merchants Bank as China Minsheng Bank; CMB, PSBC, and CZB are isolated in the banking shareholder graph.
- The trade panels contain repeated placeholder values for PSBC and CZB before their A-share listings; the forecaster masks them, and the RL panel starts on 10 December 2019.
- Shareholder lists are single snapshots published in 2023 (healthcare: as at 2022-12-31; banking: latest lists at export, June 2023). The relation graphs are static, so the 2022 and 2023 forecasting folds use graphs that contain information from their test years.

## Contact

Zhiping Chen (corresponding author), zchen@mail.xjtu.edu.cn
