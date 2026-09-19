import sys
import json
import os

sys.path.insert(0, r"G:\AI\AuK\local_train")
from gigaam_asr import transcribe


def wer(t, h):
    t, h = t.replace("ё", "е").lower().split(), h.replace("ё", "е").lower().split()
    dp = [[0] * (len(h) + 1) for _ in range(len(t) + 1)]
    for i in range(len(t) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(t) + 1):
        for j in range(1, len(h) + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (t[i - 1] != h[j - 1]))
    return dp[-1][-1] / max(1, len(t))


pack = json.load(open(r"G:\AI\AuK\local_tests\phonetic_pack\pack.json", encoding="utf-8"))
res = []
for e in pack:
    wav = os.path.join(r"G:\AI\AuK\local_tests\s3_phonetic", e["id"] + ".wav")
    hyp = transcribe(wav)
    res.append({"id": e["id"], "text": e["text"], "hyp": hyp})
json.dump(res, open(r"G:\AI\AuK\local_tests\s3_phonetic\gigaam.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
wers = [wer(r["text"], r["hyp"]) for r in res]
print("n=", len(res), "mean_wer=", round(sum(wers) / len(wers), 3))
for r, w in zip(res, wers):
    print(f"{r['id']} wer={w:.2f} hyp='{r['hyp'][:60]}'")
