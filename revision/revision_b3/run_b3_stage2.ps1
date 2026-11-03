# Plan B3, stage 2 (PROTOCOL_B3.md): waits for stage 1 and for the extended panels, then fold-2025
# forecasters, the single 2025 evaluation of the frozen policies, the reference strategies, and the analysis.
$ErrorActionPreference = 'Continue'
$pyfc = 'C:\Users\perfe\.conda\envs\tensorflow\python.exe'
$py   = 'C:\Users\perfe\.conda\envs\finrl-meta\python.exe'
$root = 'D:\paper\2\xiu'
$b3   = Join-Path $root 'revision_b3'
$rb   = Join-Path $root 'revision_backtest'
$rl   = Join-Path $b3 'b3_rl.py'
$pan  = Join-Path $root 'data2025\panels'
$logs = Join-Path $b3 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
Add-Type -Namespace B3S -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[B3S.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 (Join-Path $logs 'orchestrator2.log'); Write-Host "$(Get-Date -Format s)  $m" }
function Start-Job3($name, $exe, $wd, $argv) {
  Log "start $name"
  $p = Start-Process -FilePath $exe -ArgumentList $argv -WorkingDirectory $wd -NoNewWindow -PassThru `
         -RedirectStandardOutput (Join-Path $logs "$name.out.log") -RedirectStandardError (Join-Path $logs "$name.err.log")
  $null = $p.Handle
  return $p
}
function Wait-Job3($name, $p) { $p.WaitForExit(); Log "end $name exit=$($p.ExitCode)"; return $p.ExitCode }
Log 'stage 2 waiting for stage 1 and the extended panels'
while (-not ((Test-Path (Join-Path $logs 'orchestrator.log')) -and (Select-String -Path (Join-Path $logs 'orchestrator.log') -Pattern 'STAGE 1 DONE' -Quiet) `
        -and (Test-Path (Join-Path $pan 'bank_panel_20251231.csv')) -and (Test-Path (Join-Path $pan 'healthcare_panel_20251231.csv')))) { Start-Sleep -Seconds 60 }
Log 'stage 2 start'
# 1. fold-2025 forecasters (training t+2 <= 2023-12-31, validation 2024, test 2025) and Ridge_R
$fb = Start-Job3 'forecast_2025_bank' $pyfc $b3 @('-u','fc_run_b3.py','train','--folds','2025','--sectors','bank','--threads','9','--panel-dir',$pan)
$fh = Start-Job3 'forecast_2025_healthcare' $pyfc $b3 @('-u','fc_run_b3.py','train','--folds','2025','--sectors','healthcare','--threads','9','--panel-dir',$pan)
$cb = Wait-Job3 'forecast_2025_bank' $fb
$ch = Wait-Job3 'forecast_2025_healthcare' $fh
if ($cb -ne 0 -or $ch -ne 0) { Log 'fold-2025 forecasters failed; stopping'; exit 1 }
# 2. the single 2025 evaluation and the reference strategies
$p = Start-Job3 'rl_evaluate' $py $rb @('-u',$rl,'evaluate','--panel-dir',$pan); $null = Wait-Job3 'rl_evaluate' $p
$p = Start-Job3 'rl_references' $py $rb @('-u',$rl,'references','--panel-dir',$pan); $null = Wait-Job3 'rl_references' $p
# 3. analysis
$p = Start-Job3 'analysis' $py $b3 @('-u','b3_analysis.py',$pan,(Join-Path $b3 'results')); $null = Wait-Job3 'analysis' $p
Log 'STAGE 2 DONE'
