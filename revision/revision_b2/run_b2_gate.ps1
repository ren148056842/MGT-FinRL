# Plan B2, stage 1 (development folds 2022 and 2023 only; see PROTOCOL.md).
# Run:  powershell -ExecutionPolicy Bypass -File D:\paper\2\xiu\revision_b2\run_b2_gate.ps1
$ErrorActionPreference = 'Continue'
$pyfc = 'C:\Users\perfe\.conda\envs\tensorflow\python.exe'    # same TensorFlow 2.11 environment as v4
$root = 'D:\paper\2\xiu'
$wd   = Join-Path $root 'revision_b2'
$logs = Join-Path $wd 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 (Join-Path $logs 'orchestrator.log'); Write-Host "$(Get-Date -Format s)  $m" }
function Start-Job2($name, $argv) {
  Log "start $name"
  $p = Start-Process -FilePath $pyfc -ArgumentList $argv -WorkingDirectory $wd -NoNewWindow -PassThru `
         -RedirectStandardOutput (Join-Path $logs "$name.out.log") -RedirectStandardError (Join-Path $logs "$name.err.log")
  $null = $p.Handle
  return $p
}
function Wait-Job2($name, $p) { $p.WaitForExit(); Log "end $name exit=$($p.ExitCode)"; return $p.ExitCode }
Log "cpus=$env:NUMBER_OF_PROCESSORS"
# 1. smoke test (2 epochs, fold 2022, nothing kept)
$s = Start-Job2 'smoke' @('-u','fc_run_b2.py','smoke','--sectors','bank','--threads','9')
if ((Wait-Job2 'smoke' $s) -ne 0) { Log 'smoke failed; stopping'; exit 1 }
# 2. development folds, one process per sector
$b = Start-Job2 'gate_bank' @('-u','fc_run_b2.py','train','--folds','2022,2023','--sectors','bank','--threads','9')
$h = Start-Job2 'gate_healthcare' @('-u','fc_run_b2.py','train','--folds','2022,2023','--sectors','healthcare','--threads','9')
$cb = Wait-Job2 'gate_bank' $b
$ch = Wait-Job2 'gate_healthcare' $h
Log "GATE TRAINING DONE bank=$cb healthcare=$ch"
