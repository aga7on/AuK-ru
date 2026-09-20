# S16 gate orchestrator (фонетический буткемп).
# Главный гейт — hard-фонетический бенчмарк (frontend_probe на hard_cases_ru, 241 текст):
#   wer_mean < 0.255 (v1.0 baseline) И accent/palatalization на hard-текстах выше.
# Стоп-условия (не-регресс v1.0): TTS WER <= 0.077, first_ok >= 0.812, эмо годен >= 48.3%.
$ErrorActionPreference = 'Continue'
$AUK = 'G:\AI\AuK'
$py = "$AUK\.venv\Scripts\python.exe"
$run = "$AUK\local_train\run_s16"
$merged = "$run\merged"
$FULL = 615778297
$LOG = "$run\gate_log.txt"
Set-Location $AUK
function L($m) { Add-Content $LOG "[$(Get-Date -Format 'HH:mm:ss')] $m" }

L 's16 gate orchestrator started'

# ждём финальный чекпойнт
$deadline = (Get-Date).AddHours(3)
while (-not ((Test-Path "$run\model_6500.pt") -and (Get-Item "$run\model_6500.pt").Length -eq $FULL)) {
  if ((Get-Date) -gt $deadline) { L 'TIMEOUT waiting model_6500'; exit 1 }
  Start-Sleep -Seconds 60
}
L 'model_6500 ready'

# ждём, пока тренер полностью освободит GPU (ERRORS.MD #5)
for ($i=0; $i -lt 25; $i++) {
  $n = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'auk\.train\.train' } | Measure-Object).Count
  if ($n -eq 0) { break }
  Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 60
L 'gpu free, trainer exited'

New-Item -ItemType Directory -Force -Path $merged | Out-Null
Copy-Item "$AUK\local_train\run_s2_B\merged\config.yaml" "$merged\config.yaml" -Force

foreach ($u in @('6000','6250','6500')) {
  $CK = "$merged\auk_s16_$u.safetensors"
  if ((Test-Path "$run\model_$u.pt") -and -not (Test-Path $CK)) {
    & $py local_train\merge_lora.py --run_dir $run --ckpt "$run\model_$u.pt" --out $CK `
      --base_ckpt "$AUK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors" --lora_r 32 --lora_alpha 64 *>> $LOG
    if (Test-Path $CK) { L ("merged $u : " + (Get-Item $CK).Length) } else { L "MERGE FAILED $u" }
  }
}

$CKF = "$merged\auk_s16_6500.safetensors"
$CFG = "$merged\config.yaml"

# ГЛАВНЫЙ гейт: hard-фонетический бенчмарк (241 текст, 3 на категорию = 33)
L 'gen+eval: frontend probe (hard benchmark)'
& $py -X utf8 local_train\frontend_probe.py --n_per_cat 3 --device cuda:0 --ckpt $CKF --config $CFG --out local_tests\frontend_probe_s16 *>> $LOG
L 'gen: emotion 60'
& $py -X utf8 local_train\emotion_ru_test_s7.py --ckpt $CKF --config $CFG --out local_tests\emotion_ru_s16 --device cuda:0 *>> $LOG
L 'gen: control 120'
& $py -X utf8 local_train\run_eval_pack.py --pack local_tests\eval_pack\pack.json --ckpt $CKF --out local_tests\s16_control --device cuda:0 *>> $LOG
L 'gen: phonetic 16'
& $py -X utf8 local_train\run_eval_pack.py --pack local_tests\phonetic_pack\pack.json --ckpt $CKF --out local_tests\s16_phonetic --device cuda:0 *>> $LOG
L 'gen: seed probe'
& $py -X utf8 local_train\seed_probe_s7.py --ckpt $CKF --config $CFG --out local_tests\tmp_seed_probe_s16 --device cuda:0 *>> $LOG

L 'objective: baseline'
& $py -X utf8 local_train\baseline_report.py --pack local_tests\eval_pack\pack.json --out local_tests\s16_control --report-dir local_train\reports\deepseek_supervised\s16_report *>> $LOG

L 'manifests'
& $py -X utf8 local_train\s5_judge_manifest.py s16 s16_control s16_phonetic s16_judge_manifest.json *>> $LOG
& $py -X utf8 local_train\emotion_ru_judge_manifest_s7.py --src local_tests\emotion_ru_s16 --out local_train\reports\deepseek_supervised\emotion_s16_judge_manifest.json *>> $LOG

# произносительный прогон на hard-текстах A/B-типа (тот же протокол, что S13b)
L 'phonetics manifest on hard benchmark gens'
& $py -X utf8 local_train\s16_phonetics_manifest.py --out local_train\reports\deepseek_supervised\s16_phonetics_manifest.json *>> $LOG

$proxyDeadline = (Get-Date).AddHours(2)
$proxyUp = $false
while ((Get-Date) -lt $proxyDeadline) {
  $c = Test-NetConnection 127.0.0.1 -Port 8045 -WarningAction SilentlyContinue
  if ($c.TcpTestSucceeded) { $proxyUp = $true; L 'proxy up'; break }
  L 'waiting proxy...'; Start-Sleep -Seconds 120
}
if ($proxyUp) {
  $d = 'local_train\reports\deepseek_supervised'
  foreach ($pair in @(
    @("$d\s16_judge_manifest.json", "$d\s16_judge_results.jsonl"),
    @("$d\emotion_s16_judge_manifest.json", "$d\emotion_s16_judge_results.jsonl"),
    @("$d\s16_phonetics_manifest.json", "$d\s16_phonetics_results.jsonl")
  )) {
    if (-not (Test-Path $pair[0])) { L ("skip missing manifest " + $pair[0]); continue }
    Remove-Item $pair[1] -Force -ErrorAction SilentlyContinue
    L ("judge: " + (Split-Path $pair[1] -Leaf))
    & $py -X utf8 local_train\audit_gemini_judge_v3.py --manifest $pair[0] --out $pair[1] --workers 3 *>> $LOG
  }
} else { L 'PROXY DOWN — judges skipped' }

L 'aggregate'
& $py -X utf8 local_train\aggregate_s16.py *>> $LOG
L 'S16 GATE DONE'