# S12 gate orchestrator: ждёт model_7250 (+ ранние 6500/7000 для поиска оптимума),
# мержит, гоняет эмо-протоколы и intensity/whisper проверки, судей, агрегат.
# Урок s7: оптимум эмоций может быть в середине — оцениваем 6500, 7000, 7250.
$ErrorActionPreference = 'Continue'
$AUK = 'G:\AI\AuK'
$py = "$AUK\.venv\Scripts\python.exe"
$run = "$AUK\local_train\run_s12"
$merged = "$run\merged"
$FULL = 615778297
$LOG = "$run\gate_log.txt"
Set-Location $AUK
function L($m) { Add-Content $LOG "[$(Get-Date -Format 'HH:mm:ss')] $m" }

L 's12 gate orchestrator started'

# ждём финальный чекпойнт
$deadline = (Get-Date).AddHours(3)
while (-not ((Test-Path "$run\model_7250.pt") -and (Get-Item "$run\model_7250.pt").Length -eq $FULL)) {
  if ((Get-Date) -gt $deadline) { L 'TIMEOUT waiting model_7250'; exit 1 }
  Start-Sleep -Seconds 60
}
L 'model_7250 ready'

New-Item -ItemType Directory -Force -Path $merged | Out-Null
Copy-Item "$AUK\local_train\run_s2_B\merged\config.yaml" "$merged\config.yaml" -Force

foreach ($u in @('6500','7000','7250')) {
  $CK = "$merged\auk_s12_$u.safetensors"
  if ((Test-Path "$run\model_$u.pt") -and -not (Test-Path $CK)) {
    & $py local_train\merge_lora.py --run_dir $run --ckpt "$run\model_$u.pt" --out $CK `
      --base_ckpt "$AUK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors" --lora_r 32 --lora_alpha 64 *>> $LOG
    L ("merged $u : " + (Get-Item $CK).Length)
  }
}

$CKF = "$merged\auk_s12_7250.safetensors"
$CFG = "$merged\config.yaml"

# эмо-протокол 60 на финальном
L 'gen: emotion 60 (7250)'
& $py -X utf8 local_train\emotion_ru_test_s7.py --ckpt $CKF --config $CFG --out local_tests\emotion_ru_s12_7250 --device cuda:0 *>> $LOG
# intensity монотонность
L 'gen: intensity monotonic (7250)'
& $py -X utf8 local_train\intensity_monotonic.py --ckpt $CKF --config $CFG --out local_tests\intensity_s12_7250 --device cuda:0 *>> $LOG
# whisper проба
L 'gen: whisper probe (7250)'
& $py -X utf8 local_train\whisper_probe.py --ckpt $CKF --config $CFG --out local_tests\whisper_probe_s12 --device cuda:0 *>> $LOG
# control 120 (регресс TTS/clone)
L 'gen: control 120'
& $py -X utf8 local_train\run_eval_pack.py --pack local_tests\eval_pack\pack.json --ckpt $CKF --out local_tests\s12_7250_control --device cuda:0 *>> $LOG
L 'gen: phonetic 16'
& $py -X utf8 local_train\run_eval_pack.py --pack local_tests\phonetic_pack\pack.json --ckpt $CKF --out local_tests\s12_7250_phonetic --device cuda:0 *>> $LOG
Start-Sleep -Seconds 120  # VRAM-остывание после training val (ERRORS.MD #5)

L 'objective: baseline'
& $py -X utf8 local_train\baseline_report.py --pack local_tests\eval_pack\pack.json --out local_tests\s12_7250_control --report-dir local_train\reports\deepseek_supervised\s12_7250_report *>> $LOG

L 'manifests'
& $py -X utf8 local_train\emotion_ru_judge_manifest_s7.py --src local_tests\emotion_ru_s12_7250 --out local_train\reports\deepseek_supervised\emotion_s12_7250_judge_manifest.json *>> $LOG
& $py -X utf8 local_train\s5_judge_manifest.py s12 s12_7250_control s12_7250_phonetic s12_7250_judge_manifest.json *>> $LOG

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
    @("$d\emotion_s12_7250_judge_manifest.json", "$d\emotion_s12_7250_judge_results.jsonl"),
    @("$d\s12_7250_judge_manifest.json", "$d\s12_7250_judge_results.jsonl")
  )) {
    Remove-Item $pair[1] -Force -ErrorAction SilentlyContinue
    L ("judge: " + (Split-Path $pair[1] -Leaf))
    & $py -X utf8 local_train\audit_gemini_judge_v3.py --manifest $pair[0] --out $pair[1] --workers 3 *>> $LOG
  }
} else { L 'PROXY DOWN — judges skipped' }

L 'aggregate'
& $py -X utf8 local_train\aggregate_s12.py *>> $LOG
L 'S12 GATE DONE'