# Plan B2: ranking-aware MGT and forecast-in-state RL (pre-registered protocol)

Registered before any B2 model is trained. Nothing in this protocol is changed after results are
seen; every outcome is reported, including a failed gate.

## Motivation

In the v4 evaluation (revision v3.0), MGT forecasts entered only the 2022 training observations of
the RL agents, and the closing-return forecasts of MGT had no rank information at the tradable
horizon (return from the close of t+1 to the close of t+2) in the development years 2022 and 2023.
B2 asks whether MGT becomes useful for allocation when (i) it is trained to rank the tradable-horizon
return and (ii) its forecasts are part of the agent's state at every decision, and whether any gain
is specific to MGT rather than to a simpler forecaster.

## Data use

* Development years: 2022 and 2023 (test years of forecasting folds 2022 and 2023; RL training year
  2022 and RL validation year 2023).
* Evaluation year: 2024. It was examined in the original submission and in the v4 evaluation; this
  is stated in the manuscript. Fold-2024 forecasters are trained only after the gate below has been
  applied, and 2024 results are computed once.
* No 2025 or 2026 data are used. Relation graphs, input features, folds, scaling, architecture,
  optimiser, batch size, patience, seeds 1-3, and window are those of v4 (revision v3.0).

## Stage 1: ranking-aware forecasters

Loss (the only change to training):

    L = masked MSE over the six standardised outputs (as in v4)
        + lambda * (1 - mean over days of the cross-sectional Pearson correlation between
                        q_hat = c2_hat - c1_hat and q = c2 - c1, in log-return units),
    lambda = 1; days with fewer than five valid stocks are skipped.

q is the log return from the close of t+1 to the close of t+2, the first return a decision made at
the close of t can earn under next-day execution. Early stopping uses L on the validation year.

Models (suffix _R): A0_Full_MGT (MGT), A10_No_cross_stock, A1_Transformer_only, A2_MGNN_only,
B1_GCN_Transformer, B2_GAT_Transformer; seeds 1-3; seed-averaged forecasts are used downstream.
Ridge_R: ridge regression of q on the same 20-day window of the eight features, pooled over stocks,
penalty chosen from the v4 grid by the validation-year rank IC of q.

## Gate (development years only)

IC2 = mean daily Spearman correlation between q_hat and q over 2022 and 2023 (fold-2022 and
fold-2023 test years pooled); t = mean / sd * sqrt(days).

* G1 (signal): IC2 of MGT_R > 0 with t >= 2.0.
* G2 (MGT-specific value): IC2 of MGT_R >= IC2 of each of A10_R, A1_R and Ridge_R.

A sector passes if G1 and G2 hold. Stage 2 is run only for sectors that pass. If no sector passes,
B2 stops, the 2024 forecasts are not produced, and the result is reported as a negative development
finding.

## Stage 2: forecast-in-state RL (passing sectors only)

* State: the v4 state (252-day covariance and MACD, RSI, CCI, DX from observed prices) plus two
  forecast rows per stock: the cross-sectional z-score of q_hat_t and the predicted log range
  h1_hat - l1_hat (standardised with RL-training-year statistics). All agents have the same state
  dimension. Prices, rewards and the ledger always use observed prices.
* Forecast sources (one agent group each): Naive (q_hat = 0, range = observed log(H_t/L_t)),
  Ridge_R, A10_R, MGT_R. Forecasts are out-of-sample in every year: fold 2022 for RL training (2022),
  fold 2023 for checkpoint selection (2023), fold 2024 for evaluation (2024).
* Agents: PPO and SAC, seeds 1-10, hyperparameters and budgets of Appendix C; checkpoints at
  20-100% of the budget selected by the 2023 net Sharpe ratio; costs F1 (primary) and F0-F6.

Hypotheses (2024, F1, net Sharpe ratio, mean over the two agents and ten seeds):

* H1 (primary): MGT_R > Naive, in each passing sector.
* H2 (secondary): MGT_R > Ridge_R and MGT_R > A10_R.

Inference: paired circular block bootstrap of daily returns (block length 10; 5 and 20 as checks;
2,000 resamples; identical resampled dates for all agents and seeds); one-sided 5% level; Holm
correction across sectors for H1 and across the four H1-H2 comparisons per sector. All agents,
seeds, cost settings and forecast sources are reported.
