# Plan B3, stage 3: repair of the first RL run (see PROTOCOL_B3.md, execution note).
# 1. records for the 224 policies that finished training (done.json was not written: duplicate keyword);
# 2. retraining of the 6 TD3 tasks that failed to allocate their replay buffer (identical code and seeds);
# 3. the single 2025 evaluation, reference strategies, analysis.
param([int]$Workers = 3)
$ErrorActionPreference = 'Continue'
$py   = 'C:\Users\perfe\.conda\envs\finrl-meta\python.exe'
$root = 'D:\paper\2\xiu'
$b3   = Join-Path $root 'revision_b3'
$rb   = Join-Path $root 'revision_backtest'
$rl   = Join-Path $b3 'b3_rl.py'
$pan  = Join-Path $root 'data2025\panels'
$logs = Join-Path $b3 'logs'
Add-Type -Namespace B3R -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[B3R.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 (Join-Path $logs 'orchestrator3.log'); Write-Host "$(Get-Date -Format s)  $m" }
function Start-Job3($name, $exe, $wd, $argv) {
  Log "start $name"
  $p = Start-Process -FilePath $exe -ArgumentList $argv -WorkingDirectory $wd -NoNewWindow -PassThru `
         -RedirectStandardOutput (Join-Path $logs "$name.out.log") -RedirectStandardError (Join-Path $logs "$name.err.log")
  $null = $p.Handle
  return $p
}
function Wait-Job3($name, $p) { $p.WaitForExit(); Log "end $name exit=$($p.ExitCode)"; return $p.ExitCode }
Log 'stage 3 start'
$p = Start-Job3 'rl_finalize' $py $rb @('-u',$rl,'finalize'); if ((Wait-Job3 'rl_finalize' $p) -ne 0) { Log 'finalize failed; stopping'; exit 1 }
$p = Start-Job3 'rl_unclaim3' $py $rb @('-u',$rl,'unclaim'); $null = Wait-Job3 'rl_unclaim3' $p
$jobs = @{}
foreach ($k in 1..$Workers) { $jobs["rl_retrain_$k"] = Start-Job3 "rl_retrain_$k" $py $rb @('-u',$rl,'worker','--id',"$(100+$k)") }
foreach ($k in $jobs.Keys) { $null = Wait-Job3 $k $jobs[$k] }
$p = Start-Job3 'rl_status3' $py $rb @('-u',$rl,'status'); $null = Wait-Job3 'rl_status3' $p
$p = Start-Job3 'rl_evaluate3' $py $rb @('-u',$rl,'evaluate','--panel-dir',$pan); $null = Wait-Job3 'rl_evaluate3' $p
$p = Start-Job3 'rl_references3' $py $rb @('-u',$rl,'references','--panel-dir',$pan); $null = Wait-Job3 'rl_references3' $p
$p = Start-Job3 'analysis3' $py $b3 @('-u','b3_analysis.py',$pan,(Join-Path $b3 'results')); $null = Wait-Job3 'analysis3' $p
Log 'STAGE 3 DONE'
