# S12 early-checkpoint eval: 6500 и 7000 — эмо-протокол + control + intensity/whisper (поиск оптимума, урок s7).
$ErrorActionPreference = 'Continue'
$AUK = 'G:\AI\AuK'
$py = "$AUK\.venv\Scripts\python.exe"
$merged = "$AUK\local_train\run_s12\merged"
$LOG = "$AUK\local_train\run_s12\early_eval_log.txt"
Set-Location $AUK
function L($m) { Add-Content $LOG "[$(Get-Date -Format 'HH:mm:ss')] $m" }

foreach ($u in @('6500')) {
  $CK = "$merged\auk_s12_$u.safetensors"
  $CFG = "$merged\config.yaml"
  if (-not (Test-Path $CK)) { L "SKIP $u no merged"; continue }
  L "emotion 60 ($u)"
  & $py -X utf8 local_train\emotion_ru_test_s7.py --ckpt $CK --config $CFG --out "local_tests\emotion_ru_s12_$u" --device cuda:0 *>> $LOG
  L "control 120 ($u)"
  & $py -X utf8 local_train\run_eval_pack.py --pack local_tests\eval_pack\pack.json --ckpt $CK --out "local_tests\s12_${u}_control" --device cuda:0 *>> $LOG
  L "objective ($u)"
  & $py -X utf8 local_train\baseline_report.py --pack local_tests\eval_pack\pack.json --out "local_tests\s12_${u}_control" --report-dir "local_train\reports\deepseek_supervised\s12_${u}_report" *>> $LOG
  L "manifest+judge emotion ($u)"
  & $py -X utf8 local_train\emotion_ru_judge_manifest_s7.py --src "local_tests\emotion_ru_s12_$u" --out "local_train\reports\deepseek_supervised\emotion_s12_${u}_judge_manifest.json" *>> $LOG
  Remove-Item "local_train\reports\deepseek_supervised\emotion_s12_${u}_judge_results.jsonl" -Force -ErrorAction SilentlyContinue
  & $py -X utf8 local_train\audit_gemini_judge_v3.py --manifest "local_train\reports\deepseek_supervised\emotion_s12_${u}_judge_manifest.json" --out "local_train\reports\deepseek_supervised\emotion_s12_${u}_judge_results.jsonl" --workers 3 *>> $LOG
}
L 'EARLY EVAL DONE'