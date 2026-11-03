@echo off
title MGT-FinRL plan B3: Tushare daily prices (a few minutes) - do not close
if not exist "D:\paper\2\xiu\data2025\raw_ts" mkdir "D:\paper\2\xiu\data2025\raw_ts"
"C:\Users\perfe\.conda\envs\finrl-meta\python.exe" -u "D:\paper\2\xiu\revision_b3\b3_fetch_tushare.py" "D:\paper\2\xiu\data2025\_private\tushare_token.txt" "D:\paper\2\xiu\data2025\raw_ts" daily > "D:\paper\2\xiu\data2025\fetch_ts_daily.log" 2>&1
echo.
echo Download finished. This window can be closed.
pause
