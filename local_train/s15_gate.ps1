# S15 gate: композиция адаптеров (emotion s7@5750 α=1.0 + neutral s5@4500 α=0.5).
# Гейт: эмоции ≥48.3% И judge clone ≥8.20 И phonetics ≥8.06 И TTS не хуже v1.0.
$ErrorActionPreference = 'Continue'
$AUK = 'G:\AI\AuK'
$py = "$AUK\.venv\Scripts\python.exe"
$CK = "$AUK\local_train\run_s15\composed_e1.0_n0.5.safetensors"
$CFG = "$AUK\local_train\run_s15\config.yaml"
$LOG = "$AUK\local_train\run_s15\gate_log.txt"
Set-Location $AUK
function L($m) { Add-Content $LOG "[$(Get-Date -Format 'HH:mm:ss')] $m" }

L 's15 gate started (composed e1.0 n0.5)'
L 'gen: emotion 60'
& $py -X utf8 local_train\emotion_ru_test_s7.py --ckpt $CK --config $CFG --out local_tests\emotion_ru_s15_comp --device cuda:0 *>> $LOG
L 'gen: control 120'
& $py -X utf8 local_train\run_eval_pack.py --pack local_tests\eval_pack\pack.json --ckpt $CK --out local_tests\s15_comp_control --device cuda:0 *>> $LOG
L 'gen: phonetic 16'
& $py -X utf8 local_train\run_eval_pack.py --pack local_tests\phonetic_pack\pack.json --ckpt $CK --out local_tests\s15_comp_phonetic --device cuda:0 *>> $LOG
L 'gen: seed probe'
& $py -X utf8 local_train\seed_probe_s7.py --ckpt $CK --config $CFG --out local_tests\tmp_seed_probe_s15_comp --device cuda:0 *>> $LOG

L 'objective: baseline'
& $py -X utf8 local_train\baseline_report.py --pack local_tests\eval_pack\pack.json --out local_tests\s15_comp_control --report-dir local_train\reports\deepseek_supervised\s15_comp_report *>> $LOG

L 'manifests'
& $py -X utf8 local_train\s5_judge_manifest.py s15comp s15_comp_control s15_comp_phonetic s15_comp_judge_manifest.json *>> $LOG
& $py -X utf8 local_train\emotion_ru_judge_manifest_s7.py --src local_tests\emotion_ru_s15_comp --out local_train\reports\deepseek_supervised\emotion_s15_comp_judge_manifest.json *>> $LOG

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
    @("$d\s15_comp_judge_manifest.json", "$d\s15_comp_judge_results.jsonl"),
    @("$d\emotion_s15_comp_judge_manifest.json", "$d\emotion_s15_comp_judge_results.jsonl")
  )) {
    Remove-Item $pair[1] -Force -ErrorAction SilentlyContinue
    L ("judge: " + (Split-Path $pair[1] -Leaf))
    & $py -X utf8 local_train\audit_gemini_judge_v3.py --manifest $pair[0] --out $pair[1] --workers 3 *>> $LOG
  }
} else { L 'PROXY DOWN — judges skipped' }

L 'aggregate'
& $py -X utf8 local_train\aggregate_s15.py *>> $LOG
L 'S15 GATE DONE'