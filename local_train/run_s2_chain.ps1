# Цепочка пилотов s2: A train -> merge+eval A250/A500 -> B train -> merge+eval B250/B500.
# Мержи удаляются после эвалов (восстанавливаются пересборкой из model_*.pt за ~2 мин) — экономия диска.
# Остановка: создать файл G:\AI\_tmp\stop_s2_chain (проверяется между фазами).
$root = "G:\AI\AuK"
$log = "G:\AI\_tmp\s2_chain.log"
$stopFile = "G:\AI\_tmp\stop_s2_chain"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }
function Stopped { Test-Path $stopFile }

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:PYTHONWARNINGS = "ignore"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
$env:SPK_THREADS = "2"
Set-Location $root

$py = "$root\.venv\Scripts\python.exe"
$base = "$root\local_train\run_ru_s1\merged\auk_ru_10000.safetensors"
$pack = "$root\local_tests\eval_pack\pack.json"

function MergeEval([string]$variant, [int]$update) {
    $runDir = "$root\local_train\run_s2_$variant"
    $ckpt = "$runDir\model_$update.pt"
    $merged = "$runDir\merged\auk_s2_${variant}_${update}.safetensors"
    if (-not (Test-Path $ckpt)) { Log "[$variant u$update] checkpoint missing: $ckpt -> skip"; return }
    Log "[$variant u$update] merge start"
    & $py "$root\local_train\merge_lora.py" --run_dir $runDir --ckpt $ckpt --out $merged --base_ckpt $base *>> $log
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $merged)) { Log "[$variant u$update] MERGE FAILED"; return }
    Copy-Item "$runDir\config.yaml" "$runDir\merged\config.yaml" -Force -ErrorAction SilentlyContinue
    $out = "$root\local_tests\s2_${variant}_${update}"
    Log "[$variant u$update] eval pack -> $out"
    & $py "$root\local_train\run_eval_pack.py" --pack $pack --ckpt $merged --out $out *>> $log
    Log "[$variant u$update] eval exit=$LASTEXITCODE"
    & $py "$root\local_train\verify_eval_run.py" --pack $pack --out $out --ckpt $merged *>> $log
    & $py "$root\local_train\baseline_report.py" --pack $pack --out $out --report-dir "$root\local_train\reports\s2_${variant}_${update}" *>> $log
    Log "[$variant u$update] verify+report done; removing merged (re-merge on demand)"
    Remove-Item $merged -Force -ErrorAction SilentlyContinue
}

Log "=== S2 CHAIN START ==="
if (Stopped) { Log "stop flag at start" } else {
    Log "phase A: training (run_s2_A.ps1)"
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\local_train\run_s2_A.ps1"
    Log "phase A training done exit=$LASTEXITCODE"
    if (Stopped) { Log "stop flag: skipping A evals/B" } else {
        MergeEval "A" 250
        MergeEval "A" 500
    }
    if (-not (Stopped)) {
        Log "phase B: training (run_s2_B.ps1)"
        & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\local_train\run_s2_B.ps1"
        Log "phase B training done exit=$LASTEXITCODE"
        if (-not (Stopped)) {
            MergeEval "B" 250
            MergeEval "B" 500
        }
    }
}
Log "=== S2 CHAIN END (stopped=$(Stopped)) ==="
