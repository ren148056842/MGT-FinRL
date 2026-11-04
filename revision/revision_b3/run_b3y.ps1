# Plan B3, post-hoc addendum Y (PROTOCOL_B3.md): waits for addendum X, then forecaster depth/block variants,
# allocation baselines, regimes, market structure, PPO hyperparameter sensitivity, and the combined evaluation.
$ErrorActionPreference = 'Continue'
$pyfc = 'C:\Users\perfe\.conda\envs\tensorflow\python.exe'
$py   = 'C:\Users\perfe\.conda\envs\finrl-meta\python.exe'
$root = 'D:\paper\2\xiu'
$b3   = Join-Path $root 'revision_b3'
$rb   = Join-Path $root 'revision_backtest'
$pan  = Join-Path $root 'data2025\panels'
$logs = Join-Path $b3 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
Add-Type -Namespace B3Y -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[B3Y.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 (Join-Path $logs 'orchestrator_y.log'); Write-Host "$(Get-Date -Format s)  $m" }
function Start-Job3($name, $exe, $wd, $argv) {
  Log "start $name"
  $p = Start-Process -FilePath $exe -ArgumentList $argv -WorkingDirectory $wd -NoNewWindow -PassThru `
         -RedirectStandardOutput (Join-Path $logs "$name.out.log") -RedirectStandardError (Join-Path $logs "$name.err.log")
  $null = $p.Handle
  return $p
}
function Wait-Job3($name, $p) { $p.WaitForExit(); Log "end $name exit=$($p.ExitCode)"; return $p.ExitCode }
Log 'stage Y waiting for stage X'
while (-not ((Test-Path (Join-Path $logs 'orchestrator_x.log')) -and (Select-String -Path (Join-Path $logs 'orchestrator_x.log') -Pattern 'STAGE X DONE|stopping' -Quiet))) { Start-Sleep -Seconds 30 }
Log 'stage Y start'
$jobs = @()
$p = Start-Job3 'y_smoke2' $pyfc $b3 @('-u','b3x_ablation.py','smoke2','--threads','4','--panel-dir',$pan)
$okfc = ((Wait-Job3 'y_smoke2' $p) -eq 0)
if ($okfc) {
  foreach ($sec in @('bank','healthcare')) {
    foreach ($f in @('2022,2023','2024,2025')) {
      $n = "y_train_${sec}_$($f.Replace(',','_'))"
      $jobs += ,@($n, (Start-Job3 $n $pyfc $b3 @('-u','b3x_ablation.py','train','--folds',$f,'--sectors',$sec,'--threads','3','--variants','X5_GAT_1layer,X6_GAT_2layers,X7_Transformer_2blocks','--panel-dir',$pan)))
    }
  }
} else { Log 'depth/block smoke failed; forecaster variants skipped' }
$jobs += ,@('y_refs', (Start-Job3 'y_refs' $py $rb @('-u',(Join-Path $b3 'b3x_refs.py'),'all','--panel-dir',$pan)))
foreach ($k in 1..6) { $n = "y_ppo_$k"; $jobs += ,@($n, (Start-Job3 $n $py $rb @('-u',(Join-Path $b3 'b3x_rl.py'),'worker','--id',"$k",'--of','6','--panel-dir',$pan))) }
foreach ($j in $jobs) { $null = Wait-Job3 $j[0] $j[1] }
$p = Start-Job3 'y_ppo_analyze' $py $rb @('-u',(Join-Path $b3 'b3x_rl.py'),'analyze','--panel-dir',$pan); $null = Wait-Job3 'y_ppo_analyze' $p
$p = Start-Job3 'y_evaluate' $pyfc $b3 @('-u','b3x_ablation.py','evaluate','--panel-dir',$pan); $null = Wait-Job3 'y_evaluate' $p
Log 'STAGE Y DONE'
