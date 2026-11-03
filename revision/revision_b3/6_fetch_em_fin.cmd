@echo off
title MGT-FinRL plan B3: Eastmoney annual statements (a few minutes) - do not close
"C:\Users\perfe\.conda\envs\finrl-meta\python.exe" -u "D:\paper\2\xiu\revision_b3\b3_fetch_em_fin.py" "D:\paper\2\xiu\data2025\raw_fin" > "D:\paper\2\xiu\data2025\fetch_em_fin2.log" 2>&1
echo.
echo Download finished. This window can be closed.
pause
