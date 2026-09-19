$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\variations_v2.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "merging u7750"
& $py "local_train\merge_lora.py" --run_dir "local_train\run_ru_s1" --ckpt "local_train\run_ru_s1\model_7750.pt" --out "local_train\run_ru_s1\merged\auk_ru_7750.safetensors" --base_ckpt "G:\AI\AuK\local_train\run_ru\auk_ru_best.safetensors" --lora_r 32 --lora_alpha 64 *>> $log

Log "generating variations v2"
& $py "local_train\variation_set.py" --ckpt "G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_7750.safetensors" --out "G:\AI\AuK\local_tests\variations_v2" *>> $log

Log "judging variations v2"
& $py "local_train\judge_eval.py" --gen_dir "G:\AI\AuK\local_tests\variations_v2" --out "G:\AI\AuK\local_tests\variations_v2\results.json" *>> $log

Log "report"
& $py "local_train\variations_report.py" "G:\AI\AuK\local_tests\variations_v2" *>> $log

Remove-Item "G:\AI\_tmp\hold_controller" -Force -ErrorAction SilentlyContinue
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "G:\AI\AuK\local_train\run_controller_s1.ps1" -WindowStyle Hidden | Out-Null
Log "variations v2 done; controller restarted"
