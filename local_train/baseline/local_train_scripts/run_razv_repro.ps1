# Razv repro runner: waits for name A/B, merges u11500, reruns the val sentence in ref-mode variants.
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\razv_repro.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "waiting for name A/B to finish"
while (-not (Select-String -Path "G:\AI\_tmp\name_ab.log" -Pattern "name A/B done" -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 120
}
Log "name done; taking GPU1"
New-Item -ItemType File -Force "G:\AI\_tmp\hold_controller" | Out-Null
Start-Sleep -Seconds 20
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'auto_controller' } | ForEach-Object { taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null }

$ckpt = "local_train\run_ru_s1\model_11500.pt"
if (-not (Test-Path $ckpt)) { $ckpt = "local_train\run_ru_s1\model_last.pt" }
Log "merging $ckpt"
& $py "local_train\merge_lora.py" --run_dir "local_train\run_ru_s1" --ckpt $ckpt --out "local_train\run_ru_s1\merged\auk_ru_repro.safetensors" --base_ckpt "G:\AI\AuK\local_train\run_ru\auk_ru_best.safetensors" --lora_r 32 --lora_alpha 64 *>> $log

Log "running razv repro"
& $py "local_train\razv_repro.py" --ckpt "G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_repro.safetensors" --out "G:\AI\AuK\local_tests\razv_repro" *>> $log
& $py "local_train\judge_eval.py" --gen_dir "G:\AI\AuK\local_tests\razv_repro" --out "G:\AI\AuK\local_tests\razv_repro\results.json" *>> $log
& $py "local_train\numerals_report.py" "G:\AI\AuK\local_tests\razv_repro" *>> $log

Remove-Item "G:\AI\_tmp\hold_controller" -Force -ErrorAction SilentlyContinue
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "G:\AI\AuK\local_train\run_controller_s1.ps1" -WindowStyle Hidden | Out-Null
Log "razv repro done; controller restarted"
