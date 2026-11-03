# Plan B3: multi-relational GAT-Transformer forecasts in the RL state, evaluated on 2025 (pre-registered)

Registered before any 2025 market data are loaded by this pipeline and before any fold-2024 B3
forecaster or B3 agent is trained. Nothing below is changed after 2025 results are computed; every
outcome is reported, whatever its sign.

## 1. Why B3, and how the forecaster was chosen

Plan B2 (revision_b2/PROTOCOL.md) trained ranking-aware forecasters (suffix _R) and stopped at its
gate: the MGNN-Transformer (MGT_R) had no rank information at the tradable horizon in the
development years 2022-2023 (revision_b2/gate/GATE_RESULT.md). B3 replaces the MGNN graph encoder
by graph attention over the same three relation graphs and re-runs the downstream RL study with a
new, untouched evaluation year (2025).

The forecaster was chosen with development-year evidence only (IC2 = mean daily Spearman rank IC
of q = log(C[t+2]/C[t+1]), 2022-2023 pooled, 484 days; from gate/gate_metrics.csv):

| Model (ranking loss) | Banking IC2 (t) | Healthcare IC2 (t) |
|---|---|---|
| B2_GAT_Transformer_R (graph attention) | 0.057 (3.5) | 0.018 (1.4) |
| B1_GCN_Transformer_R (fixed GCN) | 0.054 (3.4) | 0.010 (0.8) |
| A0_Full_MGT_R (MGNN) | -0.011 (-0.8) | 0.012 (0.9) |
| A1_Transformer_only_R (no graph) | 0.070 (4.3) | 0.011 (0.9) |
| A10_No_cross_stock_R (graph layers, A = I) | 0.057 (3.6) | 0.002 (0.1) |
| Ridge_R | 0.048 (3.0) | 0.022 (1.8) |

Among the encoders that propagate information across stocks, graph attention has the highest
development IC2 in both sectors. It is therefore the forecaster of B3, called MGAT-T
(multi-relational graph-attention Transformer; code name B2_GAT_Transformer, loss of B2 with
lambda = 1). The graph-free Transformer (A1_R) ranks higher in banking; it is a registered
comparator (H2) so that any gain attributable to the relation graphs is tested, not assumed.

MGAT-T: three graph-attention layers (widths 16, 32, 64; single head; LeakyReLU slope 0.2; ELU)
whose attention is restricted to the edges of the fused relation graph A_bar = sum_r max(A_r, 0) /
max(A_r) (shareholding, investor-relations Q&A co-mention, trading-value Kendall; zero diagonal)
plus self-loops; the graph output is concatenated with the eight inputs and passed to one
Transformer encoder block per stock along the 20-day window (4 heads, feed-forward width 128,
dropout 0.1, learned positional embedding); a dense layer gives the six outputs (h1, l1, c1, h2,
l2, c2). Architecture, inputs, scaling, optimiser, batch size, patience, window and seeds 1-3 are
those of v4/B2; nothing was tuned.

## 2. Data use

| Period | Role |
|---|---|
| 2018-2021 | forecaster training only |
| 2022-2023 | development years (B2 gate; model choice above); RL training years |
| 2024 | forecaster validation of fold 2025; RL checkpoint selection |
| 2025 | evaluation year, used once |

* The relation matrices are unchanged (the three graphs of the manuscript); the Kendall graph uses
  data from 2018-01-02 to 2022-12-30, so neither 2024 nor 2025 enters any graph.
* 2024 was examined for other models in earlier versions; it serves here only for selection.
  B3 forecasts for 2024 (fold 2024) have not been produced before this commit.
* 2025 data: Tushare Pro (daily: unadjusted high, low, close and trading value; daily_basic: market
  value, PB, PS(TTM), PE(TTM); income and cash-flow statements for the TTM denominators), for the
  same 30 stocks, 2024-01-01 to 2025-12-31 (b3_fetch_tushare.py). The existing panels come from a
  terminal export that reports negative PE(TTM) and PCF; where Tushare leaves a ratio empty or does
  not provide it, the ratio is rebuilt as total market value / TTM denominator (net profit
  attributable to the parent; operating cash flow) from the statements announced by that day.
  Each ratio definition is chosen by its agreement with the existing panel over 2024 only.
* Ingestion check (before any model sees a 2025 row): over 2024, closes, highs and lows must match
  the existing panels to 1e-6 relative error and trading values to 0.5% on at least 99% of
  stock-days; otherwise the pipeline stops and the discrepancy is reported. Agreement of each ratio
  over 2024 (median absolute relative difference, sign agreement) is reported. Suspended days
  follow the convention of the existing panels (high = low = close = previous close, trading value
  '--', ratios carried forward). The extended panel = existing rows through 2024-12-31 unchanged +
  2025 rows.

## 3. Forecasters

Walk-forward folds (origin t; test = origins whose next trading day lies in the test year):

| Fold | Training (t+2 <= ) | Validation | Test | Source of runs |
|---|---|---|---|---|
| 2022 | 2020-12-31 | 2021 | 2022 | B2 (existing) |
| 2023 | 2021-12-31 | 2022 | 2023 | B2 (existing) |
| 2024 | 2022-12-31 | 2023 | 2024 | B3 |
| 2025 | 2023-12-31 | 2024 | 2025 | B3 (after ingestion) |

* Models (seeds 1-3, seed-averaged downstream): B2_GAT_Transformer_R (MGAT-T), A1_Transformer_only_R,
  B1_GCN_Transformer_R, A10_No_cross_stock_R, A0_Full_MGT_R. Loss and early stopping as in B2.
* Ridge_R (B3 version, refitted for all four folds): ridge regressions on the flattened 20-day
  window of the eight features, pooled over stocks: one for q (penalty by validation rank IC of q,
  as in B2) and one for the log range h1 - l1 (penalty by validation MSE); grid of v4.

## 4. RL with forecasts in the state

* State: v4 state (252-day covariance of daily returns; MACD, RSI, CCI, DX from observed prices)
  plus two rows: the cross-sectional z-score of q_hat_t and the predicted log range h1_hat - l1_hat,
  standardised with the 2022-2023 statistics of the same source. All agents have the same state
  dimension; prices, rewards, execution (next day) and the ledger use observed unadjusted prices.
* Forecasts are out of sample on every day: fold 2022 (2022), fold 2023 (2023), fold 2024 (2024),
  fold 2025 (2025).
* Sources: Naive (q_hat = 0, range = observed log(H_t/L_t)), Ridge_R, A1_R, MGAT-T. Agents: PPO,
  A2C, SAC, TD3, DDPG; seeds 1-5; hyperparameters and budgets of Appendix C (protocol_v4.json);
  checkpoints at 20, 40, 60, 80, 100% of the budget, selected by the 2024 net Sharpe ratio under F1.
* Ablation sources (PPO, seeds 1-5): GCN_R, A10_R, MGT_R.
* Costs: F1 (primary; 3 bp commission, 2 bp slippage, stamp duty on sales at the historical rate)
  and F0-F6 as in the manuscript.
* Non-RL references (2025): equal-weight daily rebalancing, equal-weight buy-and-hold, and top-3
  forecast-ranking rules (equal weights on the three highest q_hat, executed at the close of t+1) for
  every forecaster.

## 5. Hypotheses (2025, F1, annualised net Sharpe ratio)

Outcome per source: the daily net returns of the 25 policies (5 agents x 5 seeds), compared in
pairs with identical agent and seed.

* H1 (primary): SR(MGAT-T) > SR(Naive), in each sector; Holm across the two sectors.
* H2 (secondary): SR(MGAT-T) > SR(A1_R) and SR(MGAT-T) > SR(Ridge_R); Holm within each sector over
  H1, H2a, H2b.
* H3 (forecast level): 2025 IC2 of MGAT-T > 0 in each sector (one-sided t-test on daily ICs; Holm
  across sectors); daily IC2 differences of MGAT-T versus each other forecaster are reported with
  block-bootstrap intervals.

Inference: paired circular block bootstrap of daily returns (block length 10; 5 and 20 as checks;
2,000 resamples; identical resampled dates for all policies; centred one-sided p-values; one-sided
5% level). Descriptive: AR, SR, MDD, Calmar, volatility, turnover per agent, seed, source and cost
setting; 2024 validation results; ablation sources; non-RL references.

## 6. Commitments

* No change of forecaster, inputs, losses, architecture, sources, agents, budgets, seeds, selection
  rule, cost settings or tests after 2025 data are loaded.
* 2025 is evaluated once with the frozen policies and forecasters; reruns are allowed only to repair
  a crash, with identical code and seeds, and are logged.
* All outcomes, including negative or null ones, enter the manuscript and the response letter.

Code: revision_b3/fc_run_b3.py (forecasters), revision_b3/b3_rl.py (RL), revision_b3/b3_ingest.py
(2025 ingestion), revision_b3/b3_analysis.py (tests), revision_b3/run_b3_*.ps1 (launchers).

## Amendment 1 (2026-11-03, before any 2025 row is processed or any model sees 2025 data)

The Tushare account has access to `daily` and `daily_basic` but not to the financial statements
(`income`, `cashflow`, `balancesheet`), so negative PE(TTM) and PCF cannot be rebuilt as planned.
The four valuation ratios are therefore taken from the public Eastmoney data centre (daily valuation
report RPT_VALUEANALYSIS_DET: PE_TTM, PB_MRQ, PS_TTM, PCF_OCF_TTM; b3_fetch_em.py), which reports
negative values in the same way as the existing panels. Prices and trading values remain from Tushare
`daily`. The choice of source for each ratio follows the rule already registered: agreement with the
existing panel over 2024 (Tushare daily_basic and Eastmoney are both scored; the source with the lower
median absolute relative difference over 2024 is used for 2025). No other part of the protocol changes.

## Amendment 2 (2026-11-03, before any model sees 2025 data)

The 2024 comparison showed that the panels' PCF is not an operating-cash-flow ratio: for every stock,
total market value / PCF is constant within each calendar year and equals the fiscal-year (y - 1)
net increase in cash and cash equivalents (Eastmoney F10 annual cash-flow statement, FBCCE_ADD or
CCE_ADD; median absolute relative difference 0.0000 over 2024, sign agreement 1.00), whereas
PCF_OCF_TTM differs by about 100%. PS changes denominator at the same dates. The ingestion therefore
also scores rebuilt ratios, total market value (Eastmoney) / annual statement item of fiscal year
(y - 1): PCF with the net increase in cash; PS with total operating income or operating income
(b3_fetch_em_fin.py downloads the annual statements). The registered rule is unchanged: for each
ratio, the candidate with the lowest 2024 median absolute relative difference (coverage >= 95%) is
used for 2025.
