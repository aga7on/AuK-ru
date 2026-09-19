# AuK RU TTS supervisor: keeps training / controller / watcher alive across crashes and reboots.
$root = "G:\AI\AuK"
$log = "G:\AI\_tmp\supervisor.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }
function PyProcs($pattern) { @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match $pattern }) }
function PsProcs($pattern) { @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" | Where-Object { $_.CommandLine -match $pattern -and $_.ProcessId -ne $PID }) }

$trainAlive = (PyProcs 'auk\.train\.train').Count -gt 0
$wrapAlive = (PsProcs 'run_train_s1_resilient').Count -gt 0
$ctrlAlive = (PyProcs 'auto_controller').Count -gt 0 -or (PsProcs 'run_controller_s1').Count -gt 0
$watchAlive = (PyProcs 'watch_samples' | Where-Object { $_.CommandLine -match 'run_ru_s1' }).Count -gt 0

$stopTraining = Test-Path "G:\AI\_tmp\stop_training"
if ($stopTraining) { Log 'stop_training flag set -> training stays down' }
if (-not $stopTraining -and -not ($trainAlive -or $wrapAlive)) {
    Log 'training not running -> starting resilient wrapper'
    Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$root\local_train\run_train_s1_resilient.ps1" -WindowStyle Hidden | Out-Null
}
if (-not $ctrlAlive -and -not (Test-Path "G:\AI\_tmp\hold_controller")) {
    Log 'controller not running -> starting'
    Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$root\local_train\run_controller_s1.ps1" -WindowStyle Hidden | Out-Null
}
if (-not $watchAlive) {
    Log 'watcher not running -> starting'
    Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "& '$root\.venv\Scripts\python.exe' '$root\local_train\watch_samples.py' --samples_dir '$root\local_train\run_ru_s1\samples' --out '$root\local_train\run_ru_s1\eval_samples.jsonl' *>> 'G:\AI\_tmp\watch_s1.log'" -WindowStyle Hidden | Out-Null
}
if ($trainAlive -and $ctrlAlive -and $watchAlive) { Log 'all components alive' }

# refresh listening indexes for the latest samples
& "$root\.venv\Scripts\python.exe" "$root\local_train\make_listen_index.py" *>> "G:\AI\_tmp\listen_index.log"

# periodic disk hygiene (safe keep-lists; skips files written <10 min ago)
& "$root\local_train\cleanup.ps1"
