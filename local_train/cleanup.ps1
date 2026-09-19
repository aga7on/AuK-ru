# AuK periodic disk hygiene: prune old merged models, checkpoints, old samples and logs.
# Safe by design: protected keep-lists + skip files modified <10 min ago (may be mid-write).
$root = "G:\AI\AuK"
$log = "G:\AI\_tmp\cleanup.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }
$freed = 0
$cutoff = (Get-Date).AddMinutes(-10)

# --- 1) merged safetensors: keep protected updates + 2 newest ---
$protMerge = @(10000, 18000)
$entries = @()
foreach ($f in (Get-ChildItem "$root\local_train\run_ru_s1\merged" -Filter "auk_ru_*.safetensors" -ErrorAction SilentlyContinue)) {
    if ($f.Name -match 'auk_ru_(\d+)\.safetensors') { $entries += [pscustomobject]@{ Up = [int]$Matches[1]; File = $f } }
}
$keepLast = ($entries | Sort-Object Up | Select-Object -Last 2).Up
foreach ($e in ($entries | Sort-Object Up)) {
    if ($e.Up -in $protMerge -or $e.Up -in $keepLast) { continue }
    if ($e.File.LastWriteTime -gt $cutoff) { continue }
    $freed += $e.File.Length
    Remove-Item $e.File.FullName -Force -ErrorAction SilentlyContinue
    Log ("pruned merged: " + $e.File.Name)
}

# --- 2) checkpoints: keep protected + model_last + 2 newest ---
$protPts = @(9000, 9500, 10000, 14750, 15000, 15250, 15500, 18000)
$pts = Get-ChildItem "$root\local_train\run_ru_s1" -Filter "model_*.pt" -ErrorAction SilentlyContinue
$ptEntries = @()
foreach ($f in $pts) {
    if ($f.Name -match 'model_(\d+)\.pt') { $ptEntries += [pscustomobject]@{ Up = [int]$Matches[1]; File = $f } }
}
$keepRecent = ($ptEntries | Sort-Object Up | Select-Object -Last 2).Up
foreach ($e in ($ptEntries | Sort-Object Up)) {
    if ($e.Up -in $protPts -or $e.Up -in $keepRecent) { continue }
    if ($e.File.LastWriteTime -gt $cutoff) { continue }
    $freed += $e.File.Length
    Remove-Item $e.File.FullName -Force -ErrorAction SilentlyContinue
    Log ("pruned checkpoint: " + $e.File.Name)
}

# --- 3) old val samples (< update 10000) ---
Get-ChildItem "$root\local_train\run_ru_s1\samples" -Filter "update_*_gen.wav" -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.Name -match 'update_(\d+)_gen') {
        if ([int]$Matches[1] -lt 10000 -and $_.LastWriteTime -lt $cutoff) {
            $t = $_.FullName -replace '_gen\.wav$', '_tgt.wav'
            $freed += $_.Length
            Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
            if (Test-Path $t) { $freed += (Get-Item $t).Length; Remove-Item $t -Force -ErrorAction SilentlyContinue }
        }
    }
}

# --- 4) _tmp logs older than 3 days ---
Get-ChildItem "G:\AI\_tmp" -Filter "*.log" -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-3) } | ForEach-Object {
    $freed += $_.Length
    Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
    Log ("pruned old log: " + $_.Name)
}

if ($freed -gt 0) {
    Log ("freed {0:N2} GB; free now {1:N1} GB" -f ($freed / 1GB), ((Get-PSDrive G).Free / 1GB))
}
