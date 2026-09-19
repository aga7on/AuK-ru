# s2 technical run (20-50 updates): verifies ref loading, freeze, NaN, save/resume, generation.
# NOT for quality conclusions.
# FIXED 2026-09-16: init = u10000 merged (s1 best), NOT s0 auk_ru_best (translit model)!
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
$env:PYTHONWARNINGS = "ignore"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
$env:SPK_THREADS = "2"
Set-Location "G:\AI\AuK"
$init = "G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors"
$log = "G:\AI\_tmp\s2_tech.log"
"=== S2 TECH RUN START $(Get-Date) init=$init ===" | Out-File -Append -Encoding utf8 $log

& ".venv\Scripts\accelerate.exe" launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m auk.train.train `
    --train_jsonl "local_train\data_s2_full\train.jsonl" `
    --val_jsonl "local_train\data_s2_full\val.jsonl" `
    --config "local_train\s2_config.yaml" `
    --init_ckpt "$init" `
    --output_dir "local_train\run_s2_tech" `
    --learning_rate 1e-4 --num_train_epochs 200 --warmup_steps 20 `
    --frames_threshold 384 --max_samples 2 `
    --save_per_updates 20 --last_per_updates 20 --logging_steps 5 --val_per_updates 20 `
    --dataloader_num_workers 2 --seed 7 `
    --lora True --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --use_ema False --bf16_transformer True *>> $log
$code = $LASTEXITCODE
"=== S2 TECH RUN EXIT $code $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
