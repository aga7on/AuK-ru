# Merge S8 checkpoints (6750 / 7250 / 7750) into merged safetensors.
# Цепочка: base = run_ru_s1/merged/auk_ru_10000.safetensors (LINEAGE), r32/a64,
# config.yaml копируется из run_s2_B/merged/config.yaml (как для s5/s6/s7).
$ErrorActionPreference = 'Stop'
$py = 'G:\AI\AuK\.venv\Scripts\python.exe'
$base = 'G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors'
$cfgSrc = 'G:\AI\AuK\local_train\run_s2_B\merged\config.yaml'
$merged = 'G:\AI\AuK\local_train\run_s8\merged'
New-Item -ItemType Directory -Force -Path $merged | Out-Null

foreach ($u in @('6750','7250','7750')) {
  $ckpt = "G:\AI\AuK\local_train\run_s8\model_$u.pt"
  $out = "$merged\auk_s8_$u.safetensors"
  if (-not (Test-Path $ckpt)) { Write-Output "SKIP $u (no ckpt)"; continue }
  if (Test-Path $out) { Write-Output "SKIP $u (merged exists)"; continue }
  & $py G:\AI\AuK\local_train\merge_lora.py --run_dir G:\AI\AuK\local_train\run_s8 `
    --ckpt $ckpt --out $out --base_ckpt $base --lora_r 32 --lora_alpha 64
  if ($LASTEXITCODE -ne 0) { throw "merge failed for $u" }
  Write-Output ("MERGED {0} -> {1} ({2:N0} MB)" -f $u, $out, ((Get-Item $out).Length/1MB))
}
Copy-Item $cfgSrc "$merged\config.yaml" -Force
Write-Output ("config.yaml copied; free G: {0:N1} GB" -f ((Get-PSDrive G).Free/1GB))
