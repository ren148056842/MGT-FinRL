# Plan B3, post-hoc addendum X (PROTOCOL_B3.md): relation-layer and complete-graph ablation of MGAT,
# and the attention analysis of the registered MGAT. Runs after the registered evaluation; descriptive only.
$ErrorActionPreference = 'Continue'
$pyfc = 'C:\Users\perfe\.conda\envs\tensorflow\python.exe'
$root = 'D:\paper\2\xiu'
$b3   = Join-Path $root 'revision_b3'
$pan  = Join-Path $root 'data2025\panels'
$logs = Join-Path $b3 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
Add-Type -Namespace B3X -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[B3X.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 (Join-Path $logs 'orchestrator_x.log'); Write-Host "$(Get-Date -Format s)  $m" }
function Start-Job3($name, $exe, $wd, $argv) {
  Log "start $name"
  $p = Start-Process -FilePath $exe -ArgumentList $argv -WorkingDirectory $wd -NoNewWindow -PassThru `
         -RedirectStandardOutput (Join-Path $logs "$name.out.log") -RedirectStandardError (Join-Path $logs "$name.err.log")
  $null = $p.Handle
  return $p
}
function Wait-Job3($name, $p) { $p.WaitForExit(); Log "end $name exit=$($p.ExitCode)"; return $p.ExitCode }
Log 'stage X start'
$p = Start-Job3 'x_smoke' $pyfc $b3 @('-u','b3x_ablation.py','smoke','--threads','8','--panel-dir',$pan)
if ((Wait-Job3 'x_smoke' $p) -ne 0) { Log 'SMOKE FAILED; stopping'; exit 1 }
$jobs = @()
foreach ($sec in @('bank','healthcare')) {
  foreach ($f in @('2022,2023','2024,2025')) {
    $n = "x_train_${sec}_$($f.Replace(',','_'))"
    $jobs += ,@($n, (Start-Job3 $n $pyfc $b3 @('-u','b3x_ablation.py','train','--folds',$f,'--sectors',$sec,'--threads','4','--panel-dir',$pan)))
  }
}
$bad = 0
foreach ($j in $jobs) { if ((Wait-Job3 $j[0] $j[1]) -ne 0) { $bad += 1 } }
if ($bad -gt 0) { Log "$bad training jobs failed; stopping"; exit 1 }
$p = Start-Job3 'x_attention' $pyfc $b3 @('-u','b3x_ablation.py','attention','--panel-dir',$pan); $null = Wait-Job3 'x_attention' $p
$p = Start-Job3 'x_evaluate' $pyfc $b3 @('-u','b3x_ablation.py','evaluate','--panel-dir',$pan); $null = Wait-Job3 'x_evaluate' $p
Log 'STAGE X DONE'
