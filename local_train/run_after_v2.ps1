# Chained runner: wait for variations v2, then voices pack + stress variants on latest checkpoint.
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\after_v2.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "waiting for variations v2 to finish"
while (-not (Select-String -Path "G:\AI\_tmp\variations_v2.log" -Pattern "variations v2 done" -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 120
}
Log "variations v2 done; taking GPU1"

New-Item -ItemType File -Force "G:\AI\_tmp\hold_controller" | Out-Null
Start-Sleep -Seconds 20
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'auto_controller' } | ForEach-Object { taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null }
Log "controller stopped"

$ckpt = "local_train\run_ru_s1\model_8500.pt"
if (-not (Test-Path $ckpt)) { $ckpt = "local_train\run_ru_s1\model_last.pt" }
Log "merging $ckpt"
& $py "local_train\merge_lora.py" --run_dir "local_train\run_ru_s1" --ckpt $ckpt --out "local_train\run_ru_s1\merged\auk_ru_latest.safetensors" --base_ckpt "G:\AI\AuK\local_train\run_ru\auk_ru_best.safetensors" --lora_r 32 --lora_alpha 64 *>> $log

Log "voices pack (4 voices x 4 phrases)"
& $py "local_train\voices_pack.py" --ckpt "G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_latest.safetensors" --out "G:\AI\AuK\local_tests\voices_pack" *>> $log
& $py "local_train\judge_eval.py" --gen_dir "G:\AI\AuK\local_tests\voices_pack" --out "G:\AI\AuK\local_tests\voices_pack\results.json" *>> $log
& $py "local_train\variations_report.py" "G:\AI\AuK\local_tests\voices_pack" *>> $log

Log "stress variants (Роге marking)"
& $py "local_train\stress_variants.py" --ckpt "G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_latest.safetensors" --out "G:\AI\AuK\local_tests\stress_variants" *>> $log
& $py "local_train\judge_eval.py" --gen_dir "G:\AI\AuK\local_tests\stress_variants" --out "G:\AI\AuK\local_tests\stress_variants\results.json" *>> $log
& $py "local_train\stress_variants_report.py" *>> $log

Remove-Item "G:\AI\_tmp\hold_controller" -Force -ErrorAction SilentlyContinue
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "G:\AI\AuK\local_train\run_controller_s1.ps1" -WindowStyle Hidden | Out-Null
Log "after_v2 all done; controller restarted"
