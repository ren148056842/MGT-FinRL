@echo off
title MGT-FinRL plan B3: Eastmoney valuation download (a few minutes) - do not close
"C:\Users\perfe\.conda\envs\finrl-meta\python.exe" -u "D:\paper\2\xiu\revision_b3\b3_fetch_em.py" "D:\paper\2\xiu\data2025\raw" > "D:\paper\2\xiu\data2025\fetch_em.log" 2>&1
echo.
echo Download finished. This window can be closed.
pause
