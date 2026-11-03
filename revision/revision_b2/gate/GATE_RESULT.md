# Plan B2 gate result (development years 2022-2023)

Evaluated with gate_eval.py exactly as registered in PROTOCOL.md (commit d191256), on the
fold-2022 and fold-2023 test years. IC2 = mean daily Spearman rank IC of the tradable-horizon return
(close t+1 to close t+2); t = mean / sd * sqrt(days), 484 days.

| Sector | IC2 of MGT_R (t) | G1: IC2 > 0, t >= 2 | G2: >= A10_R / A1_R / Ridge_R | Decision |
|---|---|---|---|---|
| Banking | -0.011 (-0.78) | fail | fail / fail / fail | fail |
| Healthcare | +0.012 (0.90) | fail | pass / pass / fail | fail |

Rivals (IC2, t): banking A10_R 0.057 (3.6), A1_R 0.070 (4.3), Ridge_R 0.048 (3.0), A2_R 0.031 (2.0),
B1_GCN_R 0.054 (3.4), B2_GAT_R 0.057 (3.5); healthcare A10_R 0.002 (0.1), A1_R 0.011 (0.9),
Ridge_R 0.022 (1.8), A2_R 0.000 (0.0), B1_GCN_R 0.010 (0.8), B2_GAT_R 0.018 (1.4).
Paired daily IC differences, banking: MGT_R minus A10_R -0.068 (t -3.8), minus A1_R -0.081 (t -3.8),
minus Ridge_R -0.060 (t -2.6).

No sector passes. As registered, B2 stops here: the fold-2024 forecasters are not trained, stage 2
(forecast-in-state RL) is not run, and 2024 is not used. Full metrics: gate_metrics.csv; decision:
gate_decision.json; training logs: run_log_b2_*.csv.
