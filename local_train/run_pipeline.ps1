# Waits for dataset build to finish, runs selection + text prep, then starts training and the watcher.
$root = "G:\AI\AuK"
$log = "G:\AI\_tmp\pipeline.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Log "pipeline started; waiting for dataset metrics"
$cand = "$root\local_train\data\candidates.jsonl"
$metrics = "$root\local_train\data\metrics.jsonl"
$py = "$root\.venv\Scripts\python.exe"

$candCount = (& $py -c "print(sum(1 for _ in open(r'$cand', encoding='utf-8')))")
while ($true) {
    $n = & $py -c "print(sum(1 for _ in open(r'$metrics', encoding='utf-8')))" 2>$null
    if ($null -eq $n) { $n = 0 }
    Log "metrics: $n / $candCount"
    if ([int]$n -ge [int]$candCount) { break }
    Start-Sleep -Seconds 300
}

Log "running stage c (selection)"
& $py "$root\local_train\build_dataset.py" --stage c *>> $log
Log "running stage d (text prep)"
& $py "$root\local_train\build_dataset.py" --stage d *>> $log

$train = "$root\local_train\data\train.jsonl"
if (-not (Test-Path $train)) { Log "train.jsonl missing - abort"; exit 1 }

Log "starting training"
$p = Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$root\local_train\run_train_bg.ps1" -WindowStyle Hidden -PassThru
Log "training pid: $($p.Id)"

Log "starting sample watcher"
$w = Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "& '$py' '$root\local_train\watch_samples.py' *>> 'G:\AI\_tmp\watch_samples.log'" -WindowStyle Hidden -PassThru
Log "watcher pid: $($w.Id)"
Log "pipeline up"
