# Watchdog S16 (фонетический буткемп): от s7@5750 (update=5750) + 750 шагов -> 6500.
# Консервативно: lr 2e-6 (урок S8/S8b: 5e-6 и 3e-6 ломали first_ok), чекпойнты каждые 250
# для выбора оптимума (урок s7: оптимум был в середине, не в финале).
$ErrorActionPreference = 'Continue'
$run = 'G:\AI\AuK\local_train\run_s16'
$FULL = 615778297
$TARGET = 'model_6500.pt'

function Get-LatestGoodCkpt {
  $best = $null; $bestNum = -1
  Get-ChildItem "$run\model_*.pt" -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.Length -eq $FULL -and $_.BaseName -match '^model_(\d+)$') {
      $n = [int]$Matches[1]
      if ($n -gt $bestNum) { $bestNum = $n; $best = $_.FullName }
    }
  }
  if (-not $best) { $ml = "$run\model_last.pt"; if ((Test-Path $ml) -and (Get-Item $ml).Length -eq $FULL) { $best = $ml } }
  return ,@($best, $bestNum)
}

function Invoke-Train {
  param([int]$attempt)
  $log = "$run\train_log$($attempt+1).txt"
  Write-Output ("[watch16 $(Get-Date -Format HH:mm:ss)] attempt {0} -> {1}" -f $attempt, (Split-Path $log -Leaf))
  & G:\AI\AuK\.venv\Scripts\accelerate.exe launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m auk.train.train `
    --train_jsonl G:\AI\AuK\local_train\data_s2_full\v16b_diction_mix\train.jsonl `
    --val_jsonl G:\AI\AuK\local_train\data_s2_full\v3_s3_mix\val.jsonl `
    --config G:\AI\AuK\local_train\run_s2_B\merged\config.yaml `
    --init_ckpt G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors `
    --output_dir $run --learning_rate 2e-6 --max_updates 6500 --warmup_steps 25 `
    --frames_threshold 384 --max_samples 2 --save_per_updates 250 --logging_steps 10 `
    --val_per_updates 250 --seed 16 --lora True --lora_r 32 --lora_alpha 64 `
    --lora_dropout 0.05 --use_ema False --bf16_transformer True *> $log
  return $LASTEXITCODE
}

Set-Location G:\AI\AuK
for ($attempt = 0; $attempt -lt 6; $attempt++) {
  $done = Join-Path $run $TARGET
  if ((Test-Path $done) -and (Get-Item $done).Length -eq $FULL) { Write-Output '[watch16] 6500 done'; exit 0 }
  $g = Get-LatestGoodCkpt
  if ($g[0] -and (-not (Test-Path "$run\model_last.pt") -or (Get-Item "$run\model_last.pt").Length -ne $FULL)) {
    Copy-Item $g[0] "$run\model_last.pt" -Force
    Write-Output ("[watch16] model_last.pt <- {0}" -f (Split-Path $g[0] -Leaf))
  }
  $code = Invoke-Train $attempt
  Write-Output ("[watch16] attempt {0} exit={1}" -f $attempt, $code)
  if ((Test-Path (Join-Path $run $TARGET)) -and (Get-Item (Join-Path $run $TARGET)).Length -eq $FULL) { Write-Output '[watch16] done'; exit 0 }
  Start-Sleep -Seconds 20
}
Write-Output '[watch16] exhausted'