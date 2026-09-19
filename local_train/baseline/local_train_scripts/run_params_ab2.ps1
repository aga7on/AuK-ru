# Params A/B v2: fixed script, fresh u10000 checkpoint, hold controller during GPU use.
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\params_ab2.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "params_ab v2 start"
New-Item -ItemType File -Force "G:\AI\_tmp\hold_controller" | Out-Null
Start-Sleep -Seconds 15
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'auto_controller' } | ForEach-Object { taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null }
Log "controller stopped"

$ckpt = "local_train\run_ru_s1\model_10000.pt"
if (-not (Test-Path $ckpt)) { $ckpt = "local_train\run_ru_s1\model_last.pt" }
Log "merging $ckpt"
& $py "local_train\merge_lora.py" --run_dir "local_train\run_ru_s1" --ckpt $ckpt --out "local_train\run_ru_s1\merged\auk_ru_10000.safetensors" --base_ckpt "G:\AI\AuK\local_train\run_ru\auk_ru_best.safetensors" --lora_r 32 --lora_alpha 64 *>> $log

Log "generating params A/B (24 files)"
& $py "local_train\params_ab.py" --ckpt "G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors" --out "G:\AI\AuK\local_tests\params_ab" *>> $log

Log "judging params A/B"
& $py "local_train\judge_eval.py" --gen_dir "G:\AI\AuK\local_tests\params_ab" --out "G:\AI\AuK\local_tests\params_ab\results.json" *>> $log

Remove-Item "G:\AI\_tmp\hold_controller" -Force -ErrorAction SilentlyContinue
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "G:\AI\AuK\local_train\run_controller_s1.ps1" -WindowStyle Hidden | Out-Null
Log "params_ab v2 done; controller restarted"
