# Variations v3 on u10000: same 20 phrases x 2 seeds, direct comparison with v2 (u7750).
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\variations_v3.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "waiting for params_ab v2 to finish"
while (-not (Select-String -Path "G:\AI\_tmp\params_ab2.log" -Pattern "params_ab v2 done" -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 120
}
Log "params v2 done; taking GPU1"
New-Item -ItemType File -Force "G:\AI\_tmp\hold_controller" | Out-Null
Start-Sleep -Seconds 20
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'auto_controller' } | ForEach-Object { taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null }
Log "controller stopped; generating variations v3 on u10000"
& $py "local_train\variation_set.py" --ckpt "G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors" --out "G:\AI\AuK\local_tests\variations_v3" *>> $log
Log "judging variations v3"
& $py "local_train\judge_eval.py" --gen_dir "G:\AI\AuK\local_tests\variations_v3" --out "G:\AI\AuK\local_tests\variations_v3\results.json" *>> $log
& $py "local_train\variations_report.py" "G:\AI\AuK\local_tests\variations_v3" *>> $log
Remove-Item "G:\AI\_tmp\hold_controller" -Force -ErrorAction SilentlyContinue
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "G:\AI\AuK\local_train\run_controller_s1.ps1" -WindowStyle Hidden | Out-Null
Log "variations v3 done; controller restarted"
