# Финализация s2 (запускается после завершения run_s2_chain.ps1):
# 1) чистый u10000: контрольный сэмпл тренинговой фразы (те же текст/реф/параметры);
# 2) фонетический пак (16 заданий: ж/з, ч/ц, ы/и) на всех 5 вариантах;
# 3) слепой набор (варианты скрыты) + форма прослушивания;
# 4) сводка фактических чисел и авто-метрик;
# 5) выключение ПК (только при наличии флага G:\AI\_tmp\finalize_shutdown.flag).
$root = "G:\AI\AuK"
$log = "G:\AI\_tmp\s2_finalize.log"
$chainLog = "G:\AI\_tmp\s2_chain.log"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }
function Flagged { Test-Path "G:\AI\_tmp\finalize_shutdown.flag" }

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
$phonPack = "$root\local_tests\phonetic_pack\pack.json"

Log "=== S2 FINALIZE START (waiting for chain) ==="
$waits = 0
while ($waits -lt 720) {
    $cl = Get-Content $chainLog -Raw -ErrorAction SilentlyContinue
    if ($cl -match 'S2 CHAIN END') { break }
    Start-Sleep -Seconds 60
    $waits++
    if ($waits % 15 -eq 0) { Log "waiting for chain... ($waits min)" }
}
Log "chain finished (waits=$waits)"

# 1) чистый u10000 сэмпл
Log "step 1: clean u10000 sample"
& $py "$root\local_train\gen_u10000_sample.py" --out "$root\local_tests\phonetic_control" *>> $log
Log "u10000 sample exit=$LASTEXITCODE"

# 2) фонетический пак на всех вариантах
Log "step 2: phonetic pack"
& $py "$root\local_train\run_eval_pack.py" --pack $phonPack --ckpt $base --out "$root\local_tests\phon_u10000" *>> $log
Log "phon u10000 exit=$LASTEXITCODE"

foreach ($v in @("A", "B")) {
    foreach ($u in @(250, 500)) {
        $runDir = "$root\local_train\run_s2_$v"
        $ckpt = "$runDir\model_$u.pt"
        $merged = "$runDir\merged\auk_s2_${v}_${u}.safetensors"
        if (-not (Test-Path $ckpt)) { Log "phon $v $u : checkpoint missing -> skip"; continue }
        if (-not (Test-Path $merged)) {
            & $py "$root\local_train\merge_lora.py" --run_dir $runDir --ckpt $ckpt --out $merged --base_ckpt $base *>> $log
            Copy-Item "$runDir\config.yaml" "$runDir\merged\config.yaml" -Force -ErrorAction SilentlyContinue
        }
        if (-not (Test-Path $merged)) { Log "phon $v $u : MERGE FAILED -> skip"; continue }
        & $py "$root\local_train\run_eval_pack.py" --pack $phonPack --ckpt $merged --out "$root\local_tests\phon_${v}_${u}" *>> $log
        Log "phon $v $u exit=$LASTEXITCODE"
        Remove-Item $merged -Force -ErrorAction SilentlyContinue
    }
}

# 3) слепой набор
Log "step 3: blind set"
& $py "$root\local_train\blind_build.py" *>> $log
Log "blind exit=$LASTEXITCODE"

# 4) сводка авто-метрик по всем вариантам
Log "step 4: comparison scaffold"
& $py "$root\local_train\s2_compare_build.py" *>> $log
Log "compare exit=$LASTEXITCODE"

Log "=== S2 FINALIZE DONE ==="
if (Flagged) {
    Log "shutdown flag present -> shutting down PC in 120s"
    shutdown.exe /s /t 120 /c "s2: chain + finalize complete; shutting down (user request)"
} else {
    Log "shutdown flag absent -> PC stays on"
}
