# Stage-1 pipeline: wait for stage-0 training + metrics extension, run micro-experiment,
# select/prep dataset, then start stage-1 training and its controller/watcher.
$root = "G:\AI\AuK"
$py = "$root\.venv\Scripts\python.exe"
$log = "G:\AI\_tmp\pipeline_s1.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

function Stage0Alive {
    $procs = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -match 'auk\.train\.train' -and $_.CommandLine -match 'run_ru(?!_s1)' -and $_.CommandLine -match 'output_dir' }
    return @($procs).Count -gt 0
}

Log "pipeline s1 started; waiting for stage-0 training to end"
while (Stage0Alive) { Start-Sleep -Seconds 180 }
Log "stage-0 training ended"

$cand = "$root\local_train\data\candidates.jsonl"
$metrics = "$root\local_train\data\metrics.jsonl"
$candN = & $py -c "print(sum(1 for _ in open(r'$cand', encoding='utf-8')))" 2>$null
Log "waiting for metrics extension (target: $candN)"
$stable = 0
$last = -1
while ($true) {
    $n = & $py -c "print(sum(1 for _ in open(r'$metrics', encoding='utf-8')))" 2>$null
    if ($null -eq $n) { $n = 0 }
    $doneMarker = Select-String -Path "G:\AI\_tmp\dataset_build2.log" -Pattern "METRICS EXT DONE" -Quiet
    if ([int]$n -ge [int]$candN -or $doneMarker) {
        if ([int]$n -eq [int]$last) { $stable++; if ($stable -ge 2 -or $doneMarker) { break } } else { $stable = 0 }
    }
    $last = $n
    Log "metrics: $n / $candN"
    Start-Sleep -Seconds 120
}
Log "metrics done: $last"

Log "micro-experiment: generating cyrillic+stress with base model"
& $py "$root\local_train\eval_generate.py" --ckpt "$root\ckpts\AuK\auk_base.safetensors" --out_dir "$root\local_train\run_ru\evals\micro_cyr" --representation cyr_stress *>> $log
& $py "$root\local_train\judge_eval.py" --gen_dir "$root\local_train\run_ru\evals\micro_cyr" --out "$root\local_train\run_ru\evals\micro_cyr\results.json" *>> $log
& $py "$root\local_train\judge_eval.py" --gen_dir "$root\local_train\run_ru\evals\base_u0" --out "$root\local_train\run_ru\evals\base_u0\results_v2.json" *>> $log

$mixFile = "$root\local_train\data\stage1_mix.txt"
& $py "$root\local_train\decide_mix.py" "$root\local_train\run_ru\evals\micro_cyr\results.json" "$root\local_train\run_ru\evals\base_u0\results_v2.json" "$mixFile" *>> $log
$mix = (Get-Content $mixFile -Raw).Trim()
Log "chosen mix: $mix"

Log "selection (relaxed) + text prep"
& $py "$root\local_train\build_dataset.py" --stage c --target_clips 40000 --max_hours 30 --min_ovrl 2.9 --min_sig 2.7 *>> $log
& $py "$root\local_train\build_dataset.py" --stage d --target_clips 40000 --max_hours 30 --min_ovrl 2.9 --min_sig 2.7 --mix $mix *>> $log

$init = "$root\ckpts\AuK\auk_base.safetensors"
$best = "$root\local_train\run_ru\auk_ru_best.safetensors"
if (Test-Path $best) { $init = $best }
$env:AU_RU_INIT = $init
Log "starting stage-1 training, init=$init"
$t = Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$root\local_train\run_train_s1.ps1" -WindowStyle Hidden -PassThru
Log "train s1 pid: $($t.Id)"

Log "starting stage-1 controller"
$c = Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$root\local_train\run_controller_s1.ps1" -WindowStyle Hidden -PassThru
Log "controller s1 pid: $($c.Id)"

Log "starting stage-1 watcher"
$w = Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "& '$py' '$root\local_train\watch_samples.py' --samples_dir '$root\local_train\run_ru_s1\samples' --out '$root\local_train\run_ru_s1\eval_samples.jsonl' *>> 'G:\AI\_tmp\watch_s1.log'" -WindowStyle Hidden -PassThru
Log "watcher s1 pid: $($w.Id)"

Log "stage-1 pipeline up"
