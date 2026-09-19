$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:PYTHONWARNINGS = "ignore"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
$log = "G:\AI\_tmp\train_s1.log"
$best = "G:\AI\AuK\local_train\run_ru\auk_ru_best.safetensors"
$init = if ($env:AU_RU_INIT) { $env:AU_RU_INIT } elseif (Test-Path $best) { $best } else { "ckpts\AuK\auk_base.safetensors" }

for ($attempt = 1; $attempt -le 6; $attempt++) {
    "=== TRAIN S1 ATTEMPT $attempt START $(Get-Date) init=$init ===" | Out-File -Append -Encoding utf8 $log
    & ".venv\Scripts\accelerate.exe" launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m auk.train.train `
        --train_jsonl "local_train\data\train.jsonl" `
        --val_jsonl "local_train\data\val.jsonl" `
        --config "local_train\ru_config.yaml" `
        --init_ckpt "$init" `
        --output_dir "local_train\run_ru_s1" `
        --learning_rate 1e-4 --num_train_epochs 2 --warmup_steps 200 `
        --frames_threshold 384 --max_samples 2 `
        --save_per_updates 250 --last_per_updates 250 --logging_steps 10 --val_per_updates 250 `
        --dataloader_num_workers 2 --seed 7 `
        --lora True --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --use_ema False --bf16_transformer True *>> $log
    $code = $LASTEXITCODE
    "=== TRAIN S1 ATTEMPT $attempt EXIT $code $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
    if ($code -eq 0) { break }
    Start-Sleep -Seconds 90
}
