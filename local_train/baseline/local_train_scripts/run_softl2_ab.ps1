# Soft-sign v2 runner: hold controller, run softl2 A/B on u10000, judge, release.
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\softl2_ab.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "softl2 start"
New-Item -ItemType File -Force "G:\AI\_tmp\hold_controller" | Out-Null
Start-Sleep -Seconds 15
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'auto_controller' } | ForEach-Object { taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null }
Log "controller stopped; generating softl2"
& $py "local_train\softl2_ab.py" --ckpt "G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors" --out "G:\AI\AuK\local_tests\softl2_ab" *>> $log
& $py "local_train\judge_eval.py" --gen_dir "G:\AI\AuK\local_tests\softl2_ab" --out "G:\AI\AuK\local_tests\softl2_ab\results.json" *>> $log
& $py "local_train\numerals_report.py" "G:\AI\AuK\local_tests\softl2_ab" *>> $log
Remove-Item "G:\AI\_tmp\hold_controller" -Force -ErrorAction SilentlyContinue
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "G:\AI\AuK\local_train\run_controller_s1.ps1" -WindowStyle Hidden | Out-Null
Log "softl2 done; controller restarted"
