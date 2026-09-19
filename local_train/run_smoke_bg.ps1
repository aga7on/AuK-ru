$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONWARNINGS = "ignore"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
"=== SMOKE START $(Get-Date) ===" | Out-File -Append -Encoding utf8 "G:\AI\_tmp\smoke_train.log"
& ".venv\Scripts\accelerate.exe" launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m auk.train.train `
    --train_jsonl "local_train\smoke\train.jsonl" `
    --val_jsonl "local_train\smoke\val_bak.jsonl" `
    --config "ckpts\AuK\config.yaml" `
    --init_ckpt "ckpts\AuK\auk_base.safetensors" `
    --output_dir "local_train\smoke_out" `
    --learning_rate 1e-4 --num_train_epochs 1 --warmup_steps 2 `
    --frames_threshold 400 --max_samples 2 `
    --save_per_updates 30 --last_per_updates 30 --logging_steps 1 --val_per_updates 15 `
    --dataloader_num_workers 0 --seed 7 `
    --lora True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --use_ema False --bf16_transformer True *>> "G:\AI\_tmp\smoke_train.log"
"=== SMOKE EXIT $LASTEXITCODE $(Get-Date) ===" | Out-File -Append -Encoding utf8 "G:\AI\_tmp\smoke_train.log"
