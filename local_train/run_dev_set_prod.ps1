# Dev-set v2 (продуктовый режим, шаг 3): генерация + ru_metrics + судья (votes=1).
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\dev_set_prod.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "dev_set_prod start (product path: retries+trim+limit)"
& $py "local_train\dev_set_prod.py" *>> $log
Log "generation done; ru_metrics"
& $py "local_train\ru_metrics.py" --dir "G:\AI\AuK\local_tests\dev_set\u18000_prod" *>> $log
Log "metrics done; judge votes=1"
& $py "local_train\auto_judge.py" --dir "G:\AI\AuK\local_tests\dev_set\u18000_prod" --votes 1 *>> $log
Log "DEV_SET_PROD_ALL_DONE"
