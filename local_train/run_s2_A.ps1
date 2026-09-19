# Пилот A: только русская речь (data_s2_full v2_after_identity).
# 500 обновлений, LR 2e-5, новые optimizer/scheduler; max_updates-стоп + accounting.
# Условия зафиксированы в AB_PROTOCOL.md (A/B идентичны кроме данных).
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
$log = "G:\AI\_tmp\s2_A.log"
$data = "G:\AI\AuK\local_train\data_s2_full\v2_after_identity"

for ($attempt = 1; $attempt -le 3; $attempt++) {
    "=== S2 PILOT A ATTEMPT $attempt START $(Get-Date) init=$init ===" | Out-File -Append -Encoding utf8 $log
    & ".venv\Scripts\accelerate.exe" launch --num_processes 1 --num_machines 1 --mixed_precision bf16 -m auk.train.train `
        --train_jsonl "$data\train.jsonl" `
        --val_jsonl "$data\val.jsonl" `
        --config "local_train\s2_config.yaml" `
        --init_ckpt "$init" `
        --output_dir "local_train\run_s2_A" `
        --learning_rate 2e-5 --max_updates 500 --warmup_steps 25 `
        --frames_threshold 384 --max_samples 2 `
        --save_per_updates 250 --last_per_updates 250 --logging_steps 10 --val_per_updates 250 `
        --dataloader_num_workers 2 --seed 7 `
        --lora True --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --use_ema False --bf16_transformer True *>> $log
    $code = $LASTEXITCODE
    "=== S2 PILOT A ATTEMPT $attempt EXIT $code $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
    if ($code -eq 0) { break }
    Start-Sleep -Seconds 90
}
