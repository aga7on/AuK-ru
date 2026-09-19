# Instruct representation A/B runner: waits for softl A/B, then tests instruct text forms on u10000.
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\instruct_ab.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "waiting for softl A/B to finish"
while (-not (Select-String -Path "G:\AI\_tmp\softl_ab.log" -Pattern "softl A/B done" -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 120
}
Log "softl done; taking GPU1"
New-Item -ItemType File -Force "G:\AI\_tmp\hold_controller" | Out-Null
Start-Sleep -Seconds 20
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'auto_controller' } | ForEach-Object { taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null }
Log "controller stopped; running instruct A/B"
& $py "local_train\instruct_ab.py" --ckpt "G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors" --out "G:\AI\AuK\local_tests\instruct_ab" *>> $log
& $py "local_train\judge_eval.py" --gen_dir "G:\AI\AuK\local_tests\instruct_ab" --out "G:\AI\AuK\local_tests\instruct_ab\results.json" *>> $log
& $py "local_train\numerals_report.py" "G:\AI\AuK\local_tests\instruct_ab" *>> $log
Remove-Item "G:\AI\_tmp\hold_controller" -Force -ErrorAction SilentlyContinue
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "G:\AI\AuK\local_train\run_controller_s1.ps1" -WindowStyle Hidden | Out-Null
Log "instruct A/B done; controller restarted"
