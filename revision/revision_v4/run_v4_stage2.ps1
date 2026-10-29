# v4 experiments: walk-forward forecaster, then RL on forecast-augmented 2022 observations.
# Run:  powershell -ExecutionPolicy Bypass -File D:\paper\2\xiu\revision_v4\run_all_v4.ps1
$ErrorActionPreference = 'Continue'
$py   = 'C:\Users\perfe\.conda\envs\finrl-meta\python.exe'   # RL (Stable-Baselines3, torch)
$pyfc = 'C:\Users\perfe\.conda\envs\tensorflow\python.exe'    # forecaster (TensorFlow 2.11)
$root = 'D:\paper\2\xiu'
$fc   = Join-Path $root 'revision_v4\forecaster'
$rb   = Join-Path $root 'revision_backtest'
$logs = Join-Path $root 'revision_v4\logs'
New-Item -ItemType Directory -Force $logs | Out-Null
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 (Join-Path $logs 'orchestrator.log'); Write-Host "$(Get-Date -Format s)  $m" }
function Start-Job2($name, $exe, $wd, $argv) {
  Log "start $name"
  $p = Start-Process -FilePath $exe -ArgumentList $argv -WorkingDirectory $wd -NoNewWindow -PassThru `
         -RedirectStandardOutput (Join-Path $logs "$name.out.log") -RedirectStandardError (Join-Path $logs "$name.err.log")
  $null = $p.Handle   # cache the handle so that ExitCode is available after exit
  return $p
}
function Wait-Job2($name, $p) { $p.WaitForExit(); Log "end $name exit=$($p.ExitCode)"; return $p.ExitCode }
Log "stage 2 start (forecasts for RL already written)"
# 2. RL pipeline (audit, prepare, tests, 66 training runs, evaluation, non-RL rules), single thread
$rl = Start-Job2 'rl_all' $py $rb @('-u','run_revision_v4.py','all')
# 3. remaining forecasts (folds 2022-2024, all models, seeds 1-3) in parallel with the RL runs
$b = Start-Job2 'forecast_all_bank' $pyfc $fc @('-u','fc_run.py','train','--stage','all','--sectors','bank','--threads','9')
$h = Start-Job2 'forecast_all_healthcare' $pyfc $fc @('-u','fc_run.py','train','--stage','all','--sectors','healthcare','--threads','9')
$null = Wait-Job2 'forecast_all_bank' $b
$null = Wait-Job2 'forecast_all_healthcare' $h
$null = Wait-Job2 'rl_all' $rl
# 4. forecast-ranking rules on 2024 (needs the fold-2024 forecasts) and final report
$p = Start-Job2 'rl_nonrl_final' $py $rb @('-u','run_revision_v4.py','nonrl'); $null = Wait-Job2 'rl_nonrl_final' $p
$p = Start-Job2 'rl_report_final' $py $rb @('-u','run_revision_v4.py','report'); $null = Wait-Job2 'rl_report_final' $p
# 5. fusion-scalar initialisation sensitivity: fixed PPO, seeds 1-3, separate run directory
$env:MGT_V4_RUN = 'v4_alpha_init'; $env:MGT_V4_VARIANTS = 'Vanilla,I1,I2,I3'; $env:MGT_V4_E1 = 'none'; $env:MGT_V4_ABLATION = 'I1,I2,I3'; $env:MGT_V4_RULES = '0'
$p = Start-Job2 'rl_alpha_init' $py $rb @('-u','run_revision_v4.py','all'); $null = Wait-Job2 'rl_alpha_init' $p
Remove-Item Env:MGT_V4_RUN, Env:MGT_V4_VARIANTS, Env:MGT_V4_E1, Env:MGT_V4_ABLATION, Env:MGT_V4_RULES
Log 'ALL DONE'
