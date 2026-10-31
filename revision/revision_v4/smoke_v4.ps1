# Quick check of the v4 forecaster (2 epochs per model, fold 2022); writes nothing except the log.
$py = 'C:\Users\perfe\.conda\envs\tensorflow\python.exe'
$fc = 'D:\paper\2\xiu\revision_v4\forecaster'
$logs = 'D:\paper\2\xiu\revision_v4\logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$p = Start-Process -FilePath $py -ArgumentList @('-u', 'fc_run.py', 'smoke') -WorkingDirectory $fc -NoNewWindow -Wait -PassThru `
     -RedirectStandardOutput (Join-Path $logs 'smoke.out.log') -RedirectStandardError (Join-Path $logs 'smoke.err.log')
"smoke exit=$($p.ExitCode)" | Out-File -Append -Encoding utf8 (Join-Path $logs 'smoke.out.log')
Write-Host "smoke exit=$($p.ExitCode)"
