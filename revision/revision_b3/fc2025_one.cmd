@echo off
cd /d D:\paper\2\xiu\revision_b3
"C:\Users\perfe\.conda\envs\tensorflow\python.exe" -u fc_run_b3.py train --folds 2025 --sectors %1 --threads 4 --panel-dir D:\paper\2\xiu\data2025\panels > logs\forecast_2025_%1_early.out.log 2> logs\forecast_2025_%1_early.err.log
