# Dev-set (шаг 3): генерация 48 файлов на u18000 (raw, cuda:0) + метрики + судья (votes=1).
# Контроллер/трейнинг остаются на паузе (hold/stop флаги). UI пользователя не трогаем (cuda:1).
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$py = ".venv\Scripts\python.exe"
$log = "G:\AI\_tmp\dev_set.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "dev_set start (raw, no trim/bestofn, cuda:0)"
& $py "local_train\dev_set.py" *>> $log
Log "generation done; ru_metrics"
& $py "local_train\ru_metrics.py" --dir "G:\AI\AuK\local_tests\dev_set\u18000" *>> $log
Log "metrics done; judge votes=1"
& $py "local_train\auto_judge.py" --dir "G:\AI\AuK\local_tests\dev_set\u18000" --votes 1 *>> $log
Log "DEV_SET_ALL_DONE"
