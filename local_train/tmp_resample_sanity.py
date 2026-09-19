"""ГИПОТЕЗА: Aniemore WavLM требует 16кГц; клипы langswap 44.1кГц -> пересэмплировать и перегнать.
Проверка: тот же 100-клиповый сэмпл, resample -> 16кГц, снова gate.
"""
import json, os, torchaudio

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_tests", "tmp_resampled_16k")
os.makedirs(OUT, exist_ok=True)
rows = json.load(open(os.path.join(AUK, "local_tests", "tmp_emotion_gate_check.json"), encoding="utf-8"))
out = []
for i, r in enumerate(rows):
    dst = os.path.join(OUT, f"{i:04d}_16k.wav")
    if not os.path.exists(dst):
        wav, sr = torchaudio.load(r["file"])
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        torchaudio.save(dst, wav, 16000)
    out.append({"emotion": r["emotion"], "file": dst})
json.dump(out, open(os.path.join(AUK, "local_tests", "tmp_emotion_gate_check_16k.json"), "w", encoding="utf-8"))
print("rows:", len(out))
