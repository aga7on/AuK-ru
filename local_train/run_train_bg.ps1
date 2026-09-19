$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONWARNINGS = "ignore"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location "G:\AI\AuK"
"=== TRAIN START $(Get-Date) ===" | Out-File -Append -Encoding utf8 "G:\AI\_tmp\train_ru.log"
& ".venv\Scripts\accelerate.exe" launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m auk.train.train `
    --train_jsonl "local_train\data\train.jsonl" `
    --val_jsonl "local_train\data\val.jsonl" `
    --config "local_train\ru_config.yaml" `
    --init_ckpt "ckpts\AuK\auk_base.safetensors" `
    --output_dir "local_train\run_ru" `
    --learning_rate 1e-4 --num_train_epochs 1 --warmup_steps 100 `
    --frames_threshold 600 --max_samples 3 `
    --save_per_updates 1000 --last_per_updates 250 --logging_steps 10 --val_per_updates 250 `
    --dataloader_num_workers 2 --seed 7 `
    --lora True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --use_ema False --bf16_transformer True *>> "G:\AI\_tmp\train_ru.log"
"=== TRAIN EXIT $LASTEXITCODE $(Get-Date) ===" | Out-File -Append -Encoding utf8 "G:\AI\_tmp\train_ru.log"
