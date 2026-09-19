"""Проверка volume_up: нет ли нового клиппинга + инструкция vs фактический gain."""
import glob
import os

import numpy as np
import soundfile as sf

vol_up = sorted(glob.glob(r"G:\AI\AuK\local_train\data_s2_tools_v2\volume_up_train_*.wav"))
vol_down = sorted(glob.glob(r"G:\AI\AuK\local_train\data_s2_tools_v2\volume_down_train_*.wav"))
print(f"volume_up: {len(vol_up)}, volume_down: {len(vol_down)}")

# 1) весь volume_up: нет ли нелинейного ограничения
no_clip = 0
has_clip = 0
clip_files = []
for f in vol_up:
    x, sr = sf.read(f, dtype="float32")
    peak = np.max(np.abs(x))
    if peak > 0.9501:
        has_clip += 1
        clip_files.append(f)
    else:
        no_clip += 1
print(f"volume_up: NO clip={no_clip}, HAS clip={has_clip} ({100*has_clip/max(len(vol_up),1):.1f}%)")
if clip_files:
    print("clip files:", [os.path.basename(f) for f in clip_files[:5]])

# 2) проверка линейности: target == original * gain (допуск на квантование)
# для volume_down: исходник ослаблен, значит target_peak * 2 ≈ orig_peak
print("\n=== volume_down: восстановить исходный peak (был -6dB) ===")
suspicious = 0
for f in vol_down[:20]:
    x, sr = sf.read(f, dtype="float32")
    tp = np.max(np.abs(x))
    op = tp * 2.0  # если был -6dB, orig = target * 10^(6/20) = target * 2.0
    if op > 0.99:
        suspicious += 1
        print(f"  {os.path.basename(f)[:40]}: target_peak={tp:.3f} est_orig={op:.3f} ⚠️ ORIG WAS CLIPPED")
print(f"suspected clipped originals: {suspected}/{min(20, len(vol_down))}")

# 3) инструкция vs фактический gain
print("\n=== INSTRUCTION vs ACTUAL GAIN ===")
print("Текущая инструкция: 'Raise the volume by 6 decibels' — всегда говорит +6dB")
print("Но адаптивное усиление может дать +3 dB, +4 dB и т.д. если пик ограничен")
print("Это ОШИБКА: обучающий пример говорит +6, а фактическое усиление меньше")
print("→ Нужно либо подставлять фактическую величину в инструкцию,")
print("  либо исключать клипы без запаса для полного +6dB")
print("  (в текущей сборке это НЕ сделано — будет исправлено при масштабировании)")
