# Plan B2, stage 2 (only for sectors that passed the gate; see PROTOCOL.md and gate_decision.json).
# Usage: powershell -ExecutionPolicy Bypass -File D:\paper\2\xiu\revision_b2\run_b2_stage2.ps1 -Sectors bank,healthcare
param([string]$Sectors = 'bank,healthcare')
$ErrorActionPreference = 'Continue'
$pyfc = 'C:\Users\perfe\.conda\envs\tensorflow\python.exe'
$py   = 'C:\Users\perfe\.conda\envs\finrl-meta\python.exe'
$root = 'D:\paper\2\xiu'
$b2   = Join-Path $root 'revision_b2'
$rb   = Join-Path $root 'revision_backtest'
$logs = Join-Path $b2 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 (Join-Path $logs 'orchestrator.log'); Write-Host "$(Get-Date -Format s)  $m" }
function Start-Job2($name, $exe, $wd, $argv) {
  Log "start $name"
  $p = Start-Process -FilePath $exe -ArgumentList $argv -WorkingDirectory $wd -NoNewWindow -PassThru `
         -RedirectStandardOutput (Join-Path $logs "$name.out.log") -RedirectStandardError (Join-Path $logs "$name.err.log")
  $null = $p.Handle
  return $p
}
function Wait-Job2($name, $p) { $p.WaitForExit(); Log "end $name exit=$($p.ExitCode)"; return $p.ExitCode }
$list = $Sectors.Split(',')
Log "stage 2 sectors=$Sectors"
# 1. RL episodes for 2022 and 2023 (fold-2022 and fold-2023 forecasts only) and a check of the state construction
$p = Start-Job2 'rl_prepare' $py $rb @('-u','b2_rl.py','prepare','--sectors',$Sectors)
if ((Wait-Job2 'rl_prepare' $p) -ne 0) { Log 'prepare failed; stopping'; exit 1 }
# 2. RL training (one process per sector and algorithm) in parallel with the fold-2024 forecasters
$jobs = @{}
foreach ($s in $list) { foreach ($a in @('ppo','sac')) { $jobs["rl_${s}_${a}"] = Start-Job2 "rl_${s}_${a}" $py $rb @('-u','b2_rl.py','train','--sector',$s,'--algorithm',$a) } }
$f = Start-Job2 'forecast_2024' $pyfc $b2 @('-u','fc_run_b2.py','train','--folds','2024','--sectors',$Sectors,'--threads','4')
$null = Wait-Job2 'forecast_2024' $f
foreach ($k in $jobs.Keys) { $null = Wait-Job2 $k $jobs[$k] }
# 3. 2024 evaluation, once
$p = Start-Job2 'rl_evaluate' $py $rb @('-u','b2_rl.py','evaluate','--sectors',$Sectors)
$null = Wait-Job2 'rl_evaluate' $p
Log 'STAGE 2 DONE'
