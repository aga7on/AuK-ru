# Overnight release regression: hold controller down, run regression + flaky pack, judge, leave controller down.
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\regress_u18000.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "regress start (training paused; controller stays down)"
New-Item -ItemType File -Force "G:\AI\_tmp\hold_controller" | Out-Null
Start-Sleep -Seconds 8
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'auto_controller' } | ForEach-Object { taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null }

& $py "local_train\regress_set.py" *>> $log
Log "generation done; auto-judge"
& $py "local_train\auto_judge.py" --dir "G:\AI\AuK\local_tests\regress_u18000" --votes 1 *>> $log
Log "REGRESS_ALL_DONE (controller left down overnight)"
