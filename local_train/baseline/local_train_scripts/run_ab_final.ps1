# A/B release candidates: u18000 vs u22000 (gen + auto-judge votes=2), then release hold.
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\ab_final.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "ab_final start"
New-Item -ItemType File -Force "G:\AI\_tmp\hold_controller" | Out-Null
Start-Sleep -Seconds 8
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'auto_controller' } | ForEach-Object { taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null }
Log "controller stopped; merging + generating A/B"

& $py "local_train\ab_checkpoint.py" *>> $log
Log "generation done; auto-judge votes=2"
& $py "local_train\auto_judge.py" --dir "G:\AI\AuK\local_tests\ab_final" --votes 2 *>> $log
Log "AB done"
