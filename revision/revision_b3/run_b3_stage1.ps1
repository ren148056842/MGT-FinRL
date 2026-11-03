# Plan B3, stage 1 (no 2025 data; see PROTOCOL_B3.md): two-target Ridge_R for folds 2022-2024,
# fold-2024 forecasters, RL training on 2022-2023 with checkpoint selection on 2024.
# Run:  powershell -ExecutionPolicy Bypass -File D:\paper\2\xiu\revision_b3\run_b3_stage1.ps1
param([int]$Workers = 16)
$ErrorActionPreference = 'Continue'
$pyfc = 'C:\Users\perfe\.conda\envs\tensorflow\python.exe'    # TensorFlow 2.11 environment of v4/B2
$py   = 'C:\Users\perfe\.conda\envs\finrl-meta\python.exe'     # SB3 environment of v4
$root = 'D:\paper\2\xiu'
$b3   = Join-Path $root 'revision_b3'
$rb   = Join-Path $root 'revision_backtest'
$rl   = Join-Path $b3 'b3_rl.py'
$logs = Join-Path $b3 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
Add-Type -Namespace B3 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[B3.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null     # keep the system awake while this window is open
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 (Join-Path $logs 'orchestrator.log'); Write-Host "$(Get-Date -Format s)  $m" }
function Start-Job3($name, $exe, $wd, $argv) {
  Log "start $name"
  $p = Start-Process -FilePath $exe -ArgumentList $argv -WorkingDirectory $wd -NoNewWindow -PassThru `
         -RedirectStandardOutput (Join-Path $logs "$name.out.log") -RedirectStandardError (Join-Path $logs "$name.err.log")
  $null = $p.Handle
  return $p
}
function Wait-Job3($name, $p) { $p.WaitForExit(); Log "end $name exit=$($p.ExitCode)"; return $p.ExitCode }
Log "stage 1 cpus=$env:NUMBER_OF_PROCESSORS workers=$Workers"
# 1. Ridge_R (q and range) for folds 2022-2024
$p = Start-Job3 'ridge' $pyfc $b3 @('-u','fc_run_b3.py','ridge','--folds','2022,2023,2024','--sectors','bank,healthcare')
if ((Wait-Job3 'ridge' $p) -ne 0) { Log 'ridge failed; stopping'; exit 1 }
# 2. fold-2024 forecasters, one process per sector
$fb = Start-Job3 'forecast_2024_bank' $pyfc $b3 @('-u','fc_run_b3.py','train','--folds','2024','--sectors','bank','--threads','9')
$fh = Start-Job3 'forecast_2024_healthcare' $pyfc $b3 @('-u','fc_run_b3.py','train','--folds','2024','--sectors','healthcare','--threads','9')
$cb = Wait-Job3 'forecast_2024_bank' $fb
$ch = Wait-Job3 'forecast_2024_healthcare' $fh
if ($cb -ne 0 -or $ch -ne 0) { Log 'forecasters failed; stopping'; exit 1 }
# 3. RL episodes (2022-2024) for every source, then a short smoke run
$p = Start-Job3 'rl_prepare' $py $rb @('-u',$rl,'prepare')
if ((Wait-Job3 'rl_prepare' $p) -ne 0) { Log 'prepare failed; stopping'; exit 1 }
$p = Start-Job3 'rl_smoke' $py $rb @('-u',$rl,'smoke')
if ((Wait-Job3 'rl_smoke' $p) -ne 0) { Log 'smoke failed; stopping'; exit 1 }
# 4. RL training: workers claim tasks until none is left (longest tasks first)
$p = Start-Job3 'rl_unclaim' $py $rb @('-u',$rl,'unclaim'); $null = Wait-Job3 'rl_unclaim' $p
$jobs = @{}
foreach ($k in 1..$Workers) { $jobs["rl_worker_$k"] = Start-Job3 "rl_worker_$k" $py $rb @('-u',$rl,'worker','--id',"$k") }
foreach ($k in $jobs.Keys) { $null = Wait-Job3 $k $jobs[$k] }
$p = Start-Job3 'rl_status' $py $rb @('-u',$rl,'status'); $null = Wait-Job3 'rl_status' $p
Log 'STAGE 1 DONE'
