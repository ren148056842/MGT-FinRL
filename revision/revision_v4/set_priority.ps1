# Give the single-threaded RL process priority over the TensorFlow forecaster processes.
$out = 'D:\paper\2\xiu\revision_v4\logs\priority.txt'
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
  try {
    if ($_.Path -like '*envs\tensorflow\*') { $_.PriorityClass = 'BelowNormal' }
    elseif ($_.Path -like '*envs\finrl-meta\*') { $_.PriorityClass = 'AboveNormal' }
  } catch { }
}
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, Path, PriorityClass, CPU | Format-Table -AutoSize | Out-File -Encoding utf8 $out
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List | Out-File -Append -Encoding utf8 $out
