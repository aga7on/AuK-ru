# S8 gate orchestrator: ждёт model_7750.pt, мержит, гоняет полный GATES-прогон, судей и свод.
# Лог: local_train\run_s8\gate_log.txt. Запуск detached.
$ErrorActionPreference = 'Continue'
$AUK = 'G:\AI\AuK'
$py = "$AUK\.venv\Scripts\python.exe"
$run = "$AUK\local_train\run_s8"
$merged = "$run\merged"
$CK = "$merged\auk_s8_7750.safetensors"
$CFG = "$merged\config.yaml"
$FULL = 615778297
$LOG = "$run\gate_log.txt"
Set-Location $AUK

function L($m) { Add-Content $LOG "[$(Get-Date -Format 'HH:mm:ss')] $m" }

L 'gate orchestrator started'

# 1) wait for 7750 (max 4h)
$deadline = (Get-Date).AddHours(4)
while (-not ((Test-Path "$run\model_7750.pt") -and (Get-Item "$run\model_7750.pt").Length -eq $FULL)) {
  if ((Get-Date) -gt $deadline) { L 'TIMEOUT waiting model_7750'; exit 1 }
  Start-Sleep -Seconds 60
}
L 'model_7750 ready'

# 2) merge
if (-not (Test-Path $CK)) {
  & $py local_train\merge_lora.py --run_dir $run --ckpt "$run\model_7750.pt" --out $CK `
    --base_ckpt "$AUK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors" --lora_r 32 --lora_alpha 64 *>> $LOG
  Copy-Item "$AUK\local_train\run_s2_B\merged\config.yaml" $CFG -Force
  L ("merged 7750: " + (Get-Item $CK).Length)
}

# 3) generations (sequential, cuda:1)
L 'gen: eval pack 120'
& $py -X utf8 local_train\run_eval_pack.py --pack local_tests\eval_pack\pack.json --ckpt $CK --out local_tests\s8_7750_control --device cuda:1 *>> $LOG
L 'gen: phonetic 16'
& $py -X utf8 local_train\run_eval_pack.py --pack local_tests\phonetic_pack\pack.json --ckpt $CK --out local_tests\s8_7750_phonetic --device cuda:1 *>> $LOG
L 'gen: clone100'
& $py -X utf8 local_train\s5_clone100_gen.py --ckpt $CK --out_dir local_tests\s8_7750_clone100 --device cuda:1 *>> $LOG
L 'gen: upstream'
& $py -X utf8 local_train\upstream_control_gen.py --variant s8_7750 --ckpt $CK --out local_tests\upstream_control\s8_7750 --device cuda:1 *>> $LOG
L 'gen: emotion 60'
& $py -X utf8 local_train\emotion_ru_test_s7.py --ckpt $CK --config $CFG --out local_tests\emotion_ru_s8_7750 --device cuda:1 *>> $LOG
L 'gen: seed probe best-of-3'
& $py -X utf8 local_train\seed_probe_s7.py --ckpt $CK --config $CFG --out local_tests\tmp_seed_probe_s8_7750 --device cuda:1 *>> $LOG

# 4) objective
L 'objective: baseline_report'
& $py -X utf8 local_train\baseline_report.py --pack local_tests\eval_pack\pack.json --out local_tests\s8_7750_control --report-dir local_train\reports\deepseek_supervised\s8_7750_report *>> $LOG
L 'objective: clone100'
& $py -X utf8 local_train\s5_clone100_objective.py local_tests\s8_7750_clone100 *>> $LOG
L 'objective: dsp'
& $py -X utf8 local_train\tool_dsp_measure.py *>> $LOG
L 'objective: frontend probe (v1.0-weights-independent, на s8)'
& $py -X utf8 local_train\frontend_probe.py --n_per_cat 3 --device cuda:1 --ckpt $CK --config $CFG --out local_tests\frontend_probe_s8 *>> $LOG

# 5) manifests
L 'manifests'
& $py -X utf8 local_train\s5_judge_manifest.py s8 s8_7750_control s8_7750_phonetic s8_7750_judge_manifest.json *>> $LOG
& $py -X utf8 local_train\s5_upstream_judge_manifest.py s8_7750 s8_7750_upstream_judge_manifest.json *>> $LOG
& $py -X utf8 local_train\s5_clone100_judge_manifest.py local_tests\s8_7750_clone100 s8_7750_clone100_judge_manifest.json *>> $LOG
& $py -X utf8 local_train\emotion_ru_judge_manifest_s7.py --src local_tests\emotion_ru_s8_7750 --out local_train\reports\deepseek_supervised\emotion_s8_7750_judge_manifest.json *>> $LOG

# 6) judges (Gemini proxy must be up on 8045) — ждём прокси до 2ч, иначе судьи пропускаются
$d = 'local_train\reports\deepseek_supervised'
$proxyDeadline = (Get-Date).AddHours(2)
$proxyUp = $false
while ((Get-Date) -lt $proxyDeadline) {
  $c = Test-NetConnection 127.0.0.1 -Port 8045 -WarningAction SilentlyContinue
  if ($c.TcpTestSucceeded) { $proxyUp = $true; L 'gemini proxy up on 8045'; break }
  L 'waiting gemini proxy on 8045...'
  Start-Sleep -Seconds 120
}
if (-not $proxyUp) { L 'PROXY DOWN after 2h — judges SKIPPED (objective only)'; }

if ($proxyUp) {
foreach ($pair in @(
  @("$d\s8_7750_judge_manifest.json", "$d\s8_7750_judge_results.jsonl"),
  @("$d\s8_7750_upstream_judge_manifest.json", "$d\s8_7750_upstream_judge_results.jsonl"),
  @("$d\s8_7750_clone100_judge_manifest.json", "$d\s8_7750_clone100_judge_results.jsonl"),
  @("$d\emotion_s8_7750_judge_manifest.json", "$d\emotion_s8_7750_judge_results.jsonl")
)) {
  Remove-Item $pair[1] -Force -ErrorAction SilentlyContinue
  L ("judge: " + (Split-Path $pair[1] -Leaf))
  & $py -X utf8 local_train\audit_gemini_judge_v3.py --manifest $pair[0] --out $pair[1] --workers 3 *>> $LOG
}
}

# 7) aggregate → S8_RESULTS.md
L 'aggregate'
& $py -X utf8 local_train\aggregate_s8.py *>> $LOG
L 'GATE PIPELINE DONE'