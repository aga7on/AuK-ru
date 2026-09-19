# S10-RFT gate orchestrator: ждёт model_6500, мержит, полный прогон + свод.
# Гейт S10 (S8B_RESULTS.md → РЕШЕНИЕ): first-shot ≥0.76 + не-регресс v1.0
# (эмоции ≥48.3%, TTS WER ≤0.077, first_ok ≥0.812, clone sim best-of-3 ≥0.75).
$ErrorActionPreference = 'Continue'
$AUK = 'G:\AI\AuK'
$py = "$AUK\.venv\Scripts\python.exe"
$run = "$AUK\local_train\run_s10rft"
$merged = "$run\merged"
$CK = "$merged\auk_s10rft_6500.safetensors"
$CFG = "$merged\config.yaml"
$FULL = 615778297
$LOG = "$run\gate_log.txt"
Set-Location $AUK

function L($m) { Add-Content $LOG "[$(Get-Date -Format 'HH:mm:ss')] $m" }

L 's10 gate orchestrator started'

$deadline = (Get-Date).AddHours(3)
while (-not ((Test-Path "$run\model_6500.pt") -and (Get-Item "$run\model_6500.pt").Length -eq $FULL)) {
  if ((Get-Date) -gt $deadline) { L 'TIMEOUT waiting model_6500'; exit 1 }
  Start-Sleep -Seconds 60
}
L 'model_6500 ready'

if (-not (Test-Path $CK)) {
  New-Item -ItemType Directory -Force -Path $merged | Out-Null
  & $py local_train\merge_lora.py --run_dir $run --ckpt "$run\model_6500.pt" --out $CK `
    --base_ckpt "$AUK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors" --lora_r 32 --lora_alpha 64 *>> $LOG
  Copy-Item "$AUK\local_train\run_s2_B\merged\config.yaml" $CFG -Force
  L ("merged: " + (Get-Item $CK).Length)
}

L 'gen: eval pack 120'
& $py -X utf8 local_train\run_eval_pack.py --pack local_tests\eval_pack\pack.json --ckpt $CK --out local_tests\s10rft_control --device cuda:0 *>> $LOG
L 'gen: phonetic 16'
& $py -X utf8 local_train\run_eval_pack.py --pack local_tests\phonetic_pack\pack.json --ckpt $CK --out local_tests\s10rft_phonetic --device cuda:0 *>> $LOG
L 'gen: clone100'
& $py -X utf8 local_train\s5_clone100_gen.py --ckpt $CK --out_dir local_tests\s10rft_clone100 --device cuda:0 *>> $LOG
L 'gen: emotion 60'
& $py -X utf8 local_train\emotion_ru_test_s7.py --ckpt $CK --config $CFG --out local_tests\emotion_ru_s10rft --device cuda:0 *>> $LOG
L 'gen: seed probe'
& $py -X utf8 local_train\seed_probe_s7.py --ckpt $CK --config $CFG --out local_tests\tmp_seed_probe_s10rft --device cuda:0 *>> $LOG

L 'objective: baseline_report'
& $py -X utf8 local_train\baseline_report.py --pack local_tests\eval_pack\pack.json --out local_tests\s10rft_control --report-dir local_train\reports\deepseek_supervised\s10rft_report *>> $LOG
L 'objective: clone100'
& $py -X utf8 local_train\s5_clone100_objective.py local_tests\s10rft_clone100 *>> $LOG

L 'manifests'
& $py -X utf8 local_train\s5_judge_manifest.py s10rft s10rft_control s10rft_phonetic s10rft_judge_manifest.json *>> $LOG
& $py -X utf8 local_train\s5_clone100_judge_manifest.py local_tests\s10rft_clone100 s10rft_clone100_judge_manifest.json *>> $LOG
& $py -X utf8 local_train\emotion_ru_judge_manifest_s7.py --src local_tests\emotion_ru_s10rft --out local_train\reports\deepseek_supervised\emotion_s10rft_judge_manifest.json *>> $LOG

$d = 'local_train\reports\deepseek_supervised'
$proxyDeadline = (Get-Date).AddHours(2)
$proxyUp = $false
while ((Get-Date) -lt $proxyDeadline) {
  $c = Test-NetConnection 127.0.0.1 -Port 8045 -WarningAction SilentlyContinue
  if ($c.TcpTestSucceeded) { $proxyUp = $true; L 'gemini proxy up'; break }
  L 'waiting gemini proxy...'
  Start-Sleep -Seconds 120
}
if (-not $proxyUp) { L 'PROXY DOWN — judges SKIPPED' }

if ($proxyUp) {
foreach ($pair in @(
  @("$d\s10rft_judge_manifest.json", "$d\s10rft_judge_results.jsonl"),
  @("$d\s10rft_clone100_judge_manifest.json", "$d\s10rft_clone100_judge_results.jsonl"),
  @("$d\emotion_s10rft_judge_manifest.json", "$d\emotion_s10rft_judge_results.jsonl")
)) {
  Remove-Item $pair[1] -Force -ErrorAction SilentlyContinue
  L ("judge: " + (Split-Path $pair[1] -Leaf))
  & $py -X utf8 local_train\audit_gemini_judge_v3.py --manifest $pair[0] --out $pair[1] --workers 3 *>> $LOG
}
}

L 'aggregate'
& $py -X utf8 local_train\aggregate_s10.py *>> $LOG
L 'S10 GATE PIPELINE DONE'