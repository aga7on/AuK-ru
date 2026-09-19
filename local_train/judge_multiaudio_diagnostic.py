"""Small multi-audio diagnostic for the Gemini judge (no 685 re-run).

Checks whether the judge (a) transcribes two DIFFERENT known clips independently and in order,
(b) distinguishes same vs different speakers, (c) reports leak relative to a target text.
Writes JUDGE_DIAGNOSTIC.md. Phonetic minimal-pair sensitivity is NOT provable here and is logged
as unproven.
"""
import base64
import json
import os
import time
import urllib.request

from gemini_creds import get_key, get_model, get_url

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
REP = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
OUT = os.path.join(REP, "JUDGE_DIAGNOSTIC.md")
URL, KEY, MODEL = get_url(), get_key(), get_model()

P1 = os.path.join(AUK, "local_train", "run_s2_A", "samples", "update_500_tgt.wav")   # "Сбер, включи музыку..."
P2 = os.path.join(LT, "phon_u10000", "ph00_user_male.wav")                           # "Жёлтый жук жужжит..."
SAME_A = os.path.join(LT, "user_ref.wav")
SAME_B = os.path.join(LT, "user_ref.wav")
DIFF_A = os.path.join(LT, "user_ref.wav")
DIFF_B = os.path.join(LT, "ref_female2.wav")


def call(wavs, prompt):
    content = [{"type": "input_audio", "input_audio": {
        "data": base64.b64encode(open(p, "rb").read()).decode(), "format": "wav"}} for p in wavs]
    content.append({"type": "text", "text": prompt})
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": 600, "temperature": 0.0}
    for _ in range(3):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                         headers={"Authorization": "Bearer " + KEY,
                                                  "Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
            c = d["choices"][0]["message"].get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            t = str(c)
            m, n = t.find("{"), t.rfind("}")
            return (json.loads(t[m:n + 1]) if m >= 0 and n > m else None), t[:500]
        except Exception as e:
            time.sleep(3)
            last = f"{type(e).__name__}: {e}"
    return None, last


def main():
    tests = []
    tr_prompt = ("Ты слышишь два аудио. Транскрибируй КАЖДОЕ отдельно, дословно, по-русски. "
                 'Ответь ТОЛЬКО JSON: {"audio1":"...","audio2":"..."}')
    o1, r1 = call([P1, P2], tr_prompt)
    o2, r2 = call([P2, P1], tr_prompt)
    spk = ('Слышишь два аудио. Один и тот же это говорящий или разные? '
           'Ответь ТОЛЬКО JSON: {"same_speaker": bool, "reason":"..."}')
    o3, r3 = call([SAME_A, SAME_B], spk)
    o4, r4 = call([DIFF_A, DIFF_B], spk)
    leak = ('АУДИО 1 — референс, АУДИО 2 — синтез. Целевой текст синтеза: "Сбер, включи музыку '
            'для пробежек через десять минут". Ответь ТОЛЬКО JSON: '
            '{"ref_content_leak": bool, "leaked_ref_content": []} — есть ли в АУДИО 2 лишнее '
            'содержимое из АУДИО 1, отсутствующее в целевом тексте.')
    o5, r5 = call([os.path.join(LT, "ref_ru.wav"), P1], leak)

    def norm(s):
        return (s or "").lower()

    a1 = norm((o1 or {}).get("audio1"))
    a2 = norm((o1 or {}).get("audio2"))
    b1 = norm((o2 or {}).get("audio1"))
    b2 = norm((o2 or {}).get("audio2"))
    sber = "сбер"
    zhuk = "жук"
    ok1 = sber in a1 and zhuk in a2
    ok2 = zhuk in b1 and sber in b2
    tests.append(("order_identification", bool(ok1 and ok2),
                  {"P1,P2": (a1[:60], a2[:60]), "P2,P1": (b1[:60], b2[:60])}))
    tests.append(("same_speaker_control", (o3 or {}).get("same_speaker") is True, o3))
    tests.append(("different_speaker_control", (o4 or {}).get("same_speaker") is False, o4))
    tests.append(("leak_report_present", o5 is not None, o5))

    L = ["# Диагностика автосудьи: два аудио, порядок, говорящий (16.09.2026)", "",
         f"Модель `{MODEL}`, prompt `diagnostic-1`. Файлы: P1=update_500_tgt (Сбер-фраза),",
         "P2=ph00 (Жёлтый жук), same=(user_ref,user_ref), diff=(user_ref,ref_female2).", "",
         "| тест | результат | детали |", "|---|---|---|"]
    for name, ok, det in tests:
        L.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {json.dumps(det, ensure_ascii=False)[:300]} |")
    L += ["", "## Сырые ответы", ""]
    for name, raw in [("order(P1,P2)", r1), ("order(P2,P1)", r2), ("same", r3), ("diff", r4), ("leak", r5)]:
        L.append(f"### {name}")
        L.append("```")
        L.append(str(raw)[:700])
        L.append("```")
    L += ["", "## Ограничение", "",
          "- Порядок и различимость говорящих проверены на известных разных клипах.",
          "- **Фонетическая чувствительность к минимальным парам (ж/з, ч/ц, ы/и) здесь НЕ доказана**:",
          "  нет верифицированного ground-truth стимула. Считать её недоказанной, а `phoneme_substitutions`",
          "  — вспомогательным, непроверенным сигналом; высокая чувствительность к контрасту слов не тестировалась.",
          "- `audio_access=true` — самоотчёт, не доказательство доступа к аудио; доказательство даёт только",
          "  правильная транскрипция/различение выше.", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written", OUT)
    for name, ok, det in tests:
        print(("PASS" if ok else "FAIL"), name, json.dumps(det, ensure_ascii=False)[:200])


if __name__ == "__main__":
    main()
