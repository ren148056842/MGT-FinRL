$ErrorActionPreference = 'Stop'
$revisionPython = 'C:\Users\perfe\.conda\envs\finrl-meta\python.exe'
& $revisionPython (Join-Path $PSScriptRoot 'run_revision_v3.py') all
if ($LASTEXITCODE -ne 0) { throw "V3 failed with exit code $LASTEXITCODE" }
