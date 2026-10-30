#!/bin/sh
# Reproduces the forecasting and portfolio tables and figures of the revised manuscript (v3.1) from the stored outputs.
# Run from the repository root:  sh revision/analysis_v4/run_analysis.sh
set -e
REV=revision; A=$REV/analysis_v4; T=$(mktemp -d)
mkdir -p $T/v3 $T/v4 $T/alpha
tar xzf $REV/revision_backtest/runs/v3_train_only/results/evaluations.tar.gz -C $T/v3
tar xzf $REV/revision_backtest/runs/v4_forecast_train_only/results/evaluations.tar.gz -C $T/v4
tar xzf $REV/revision_backtest/runs/v4_forecast_train_only/results/nonrl.tar.gz -C $T/v4
tar xzf $REV/revision_backtest/runs/v4_alpha_init/results/evaluations.tar.gz -C $T/alpha
# forecasting evaluation (Tables 2, 10, 11) and loss curves (Fig. 6); the trade panels are in revision/<sector folders>
python $A/fc_eval.py $REV $REV/revision_v4/forecasts $A/results/forecast
python $A/fc_tables.py $A/results/forecast $A/results
python $A/fig_fc_loss.py $REV/revision_v4/forecasts $A/results/fig_fc_loss.pdf
# portfolio evaluation (Tables 3-7, Fig. 5)
python $A/rl_eval.py $T/v3/evaluations $T/v4/evaluations $T/v4/nonrl $T/alpha/evaluations x $A/results/rl
python $A/rules_eval.py $T/v4/nonrl $A/results/forecast $A/results/rl
python $A/ablation_boot.py $T/v3/evaluations $T/v4/evaluations $T/alpha/evaluations $A/results/rl/ablation_ppo_bootstrap.csv
python $A/tables_v4.py $A/results/rl $A/results
python $A/retro_diag.py $A/results/rl $A/results    # Appendix E, Table 14
python $A/fig_nav_v4.py $REV $T/v3/evaluations $T/v4/evaluations $A/results/fig_nav_2024.pdf
