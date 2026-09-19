# Best-of-N demo runner: hold controller, generate 5 phrases x 3 seeds, ASR-pick, release.
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\bestofn.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "bestofn start"
New-Item -ItemType File -Force "G:\AI\_tmp\hold_controller" | Out-Null
Start-Sleep -Seconds 15
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'auto_controller' } | ForEach-Object { taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null }
Log "controller stopped; generating"
& $py "local_train\bestofn_gen.py" --ckpt "G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_13400.safetensors" --out "G:\AI\AuK\local_tests\bestofn" *>> $log
Log "generation done; scoring"
& $py "local_train\bestofn_pick.py" --dir "G:\AI\AuK\local_tests\bestofn" *>> $log
Remove-Item "G:\AI\_tmp\hold_controller" -Force -ErrorAction SilentlyContinue
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "G:\AI\AuK\local_train\run_controller_s1.ps1" -WindowStyle Hidden | Out-Null
Log "bestofn done; controller restarted"
