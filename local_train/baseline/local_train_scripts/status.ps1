# AuK RU-LoRA pipeline status
$py = "G:\AI\AuK\.venv\Scripts\python.exe"
$root = "G:\AI\AuK\local_train"

Write-Output "===== AuK RU-LoRA status: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="

Write-Output "`n-- processes --"
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "build_dataset|auk\.train\.train|watch_samples" -and $_.Name -match "python" }
if ($procs) { $procs | ForEach-Object { "PID {0,-7} {1}" -f $_.ProcessId, ($_.CommandLine -replace '.*(build_dataset|auk\.train\.train|watch_samples).*', '$1') } } else { "no pipeline processes running" }

Write-Output "`n-- dataset --"
foreach ($f in @("candidates.jsonl", "metrics.jsonl", "selected.jsonl", "train.jsonl", "val.jsonl")) {
    $p = "$root\data\$f"
    if (Test-Path $p) {
        $n = & $py -c "print(sum(1 for _ in open(r'$p', encoding='utf-8')))" 2>$null
        "{0,-18} {1} lines" -f $f, $n
    }
}

Write-Output "`n-- training log (train_ru) --"
$tl = "G:\AI\_tmp\train_ru.log"
if (Test-Path $tl) {
    Get-Content $tl -Encoding Unicode | Select-String -Pattern "TRAIN START|TRAIN EXIT|resumed|val loss per-t" | Select-Object -Last 4 | ForEach-Object { $_.Line }
    Get-Content $tl -Encoding Unicode | Select-String -Pattern "\[epoch .* update .*\]" | Select-Object -Last 3 | ForEach-Object { $_.Line }
} else { "no train log yet" }

Write-Output "`n-- smoke run --"
$sl = "G:\AI\_tmp\smoke_train.log"
if (Test-Path $sl) {
    Get-Content $sl -Encoding Unicode | Select-String -Pattern "SMOKE START|SMOKE EXIT|\[update" | Select-Object -Last 3 | ForEach-Object { $_.Line }
}

Write-Output "`n-- sample evaluation (watcher) --"
$el = "$root\run_ru\eval_samples.jsonl"
if (Test-Path $el) {
    Get-Content $el -Encoding UTF8 | Select-Object -Last 5 | ForEach-Object {
        $r = $_ | ConvertFrom-Json
        "u{0,-5} gen ovrl={1,-5} f0={2,-6} dur={3,-5} | tgt ovrl={4,-5} f0={5,-6} | ratio={6}" -f $r.update, $r.gen_metrics.ovrl, $r.gen_metrics.f0, $r.gen_metrics.dur, $r.tgt_metrics.ovrl, $r.tgt_metrics.f0, $r.dur_ratio
    }
} else { "no evaluated samples yet" }

Write-Output "`n-- controller (deep evals / decisions) --"
$cl = "$root\run_ru\control.jsonl"
if (Test-Path $cl) {
    Get-Content $cl -Encoding UTF8 | Select-Object -Last 4 | ForEach-Object {
        $r = $_ | ConvertFrom-Json
        "u{0,-5} [{1}] score={2,-6} overall={3} accent={4} nat={5} sim={6} text={7}/6 gender={8}/6" -f $r.update, $r.label, $r.score, $r.means.overall, $r.means.accent, $r.means.naturalness, $r.means.voice_similarity, $r.means.text_matches, $r.means.same_gender
    }
} else { "no controller evals yet" }
foreach ($f in @("STOP_REASON.txt", "BEST.txt")) {
    $p = "$root\run_ru\$f"
    if (Test-Path $p) { "--- $f ---"; Get-Content $p -Encoding UTF8 | Select-Object -First 4 }
}
if (Test-Path "$root\run_ru\auk_ru_best.safetensors") { "BEST checkpoint: run_ru\auk_ru_best.safetensors" }

Write-Output "`n-- stage-1 pipeline --"
$pl = "G:\AI\_tmp\pipeline_s1.log"
if (Test-Path $pl) { Get-Content $pl -Encoding UTF8 | Select-String -Pattern "pipeline|metrics|micro|mix|selection|starting|pid" | Select-Object -Last 5 | ForEach-Object { $_.Line } }
$s1 = "$root\run_ru_s1"
if (Test-Path "$s1\control.jsonl") {
    Get-Content "$s1\control.jsonl" -Encoding UTF8 | Select-Object -Last 3 | ForEach-Object {
        $r = $_ | ConvertFrom-Json
        "s1 u{0,-5} [{1}] score={2,-6} overall={3} accent={4} nat={5} text={6}/6" -f $r.update, $r.label, $r.score, $r.means.overall, $r.means.accent, $r.means.naturalness, $r.means.text_matches
    }
}
$t1 = "G:\AI\_tmp\train_s1.log"
if (Test-Path $t1) {
    Get-Content $t1 -Encoding Unicode | Select-String -Pattern "TRAIN S1 START|TRAIN S1 EXIT" | ForEach-Object { $_.Line }
    Get-Content $t1 -Encoding Unicode | Select-String -Pattern "\[epoch .* update .*\]" | Select-Object -Last 2 | ForEach-Object { $_.Line }
}

Write-Output "`n-- checkpoints --"
if (Test-Path "$root\run_ru") {
    Get-ChildItem "$root\run_ru\model_*.pt" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 3 | ForEach-Object { "  {0} ({1:N0} MB)" -f $_.Name, ($_.Length / 1MB) }
} else { "run_ru dir not created yet" }

Write-Output "`n-- GPU --"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
