"""Слепой набор для прослушивания: 5 вариантов (u10000, A@250, A@500, B@250, B@500),
вариант скрыт. Включает: 120 основных заданий + 16 фонетических (перенос ж/з, ч/ц, ы/и)
+ 5 сэмплов тренинговой фразы.

Выход: G:\\AI\\AuK\\local_tests\\blind_s2\\
  wav\\NNNN.wav        - файлы без метки варианта (перемешаны)
  LISTEN.csv          - file, order, task_id, group
  listen_form.csv     - форма (пустая): text_accuracy,pronunciation,naturalness,voice_similarity,notes
  SECRET_map.csv      - file -> variant (не открывать до окончания прослушивания)
  README.md
"""
import csv
import json
import os
import random
import shutil

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
OUT = os.path.join(LT, "blind_s2")
SEED = 16

VARIANTS = ["u10000", "A@250", "A@500", "B@250", "B@500"]
MAIN_DIRS = {
    "u10000": os.path.join(LT, "u0_control"),
    "A@250": os.path.join(LT, "s2_A_250"),
    "A@500": os.path.join(LT, "s2_A_500"),
    "B@250": os.path.join(LT, "s2_B_250"),
    "B@500": os.path.join(LT, "s2_B_500"),
}
PHON_DIRS = {
    "u10000": os.path.join(LT, "phon_u10000"),
    "A@250": os.path.join(LT, "phon_A_250"),
    "A@500": os.path.join(LT, "phon_A_500"),
    "B@250": os.path.join(LT, "phon_B_250"),
    "B@500": os.path.join(LT, "phon_B_500"),
}
TRAIN_SAMPLES = {
    "u10000": os.path.join(LT, "phonetic_control", "update_u10000_gen.wav"),
    "A@250": os.path.join(AUK, "local_train", "run_s2_A", "samples", "update_250_gen.wav"),
    "A@500": os.path.join(AUK, "local_train", "run_s2_A", "samples", "update_500_gen.wav"),
    "B@250": os.path.join(AUK, "local_train", "run_s2_B", "samples", "update_250_gen.wav"),
    "B@500": os.path.join(AUK, "local_train", "run_s2_B", "samples", "update_500_gen.wav"),
}


def main():
    pack = json.load(open(os.path.join(LT, "eval_pack", "pack.json"), encoding="utf-8"))
    phon = json.load(open(os.path.join(LT, "phonetic_pack", "pack.json"), encoding="utf-8"))
    tasks = [(p["id"], p["kind"]) for p in pack] + [(p["id"], "phonetics") for p in phon]

    os.makedirs(os.path.join(OUT, "wav"), exist_ok=True)
    items = []
    for variant in VARIANTS:
        for task_id, group in tasks:
            src = os.path.join(MAIN_DIRS[variant] if group != "phonetics" else PHON_DIRS[variant],
                               f"{task_id}.wav")
            if os.path.exists(src):
                items.append((variant, task_id, group, src))
        ts = TRAIN_SAMPLES.get(variant)
        if ts and os.path.exists(ts):
            items.append((variant, "phrase_sber", "phrase_training", ts))

    random.Random(SEED).shuffle(items)
    listen_rows, form_rows, secret_rows = [], [], []
    for i, (variant, task_id, group, src) in enumerate(items, 1):
        name = f"{i:04d}.wav"
        shutil.copy2(src, os.path.join(OUT, "wav", name))
        listen_rows.append({"file": name, "order": i, "task_id": task_id, "group": group})
        form_rows.append({"file": name, "text_accuracy": "", "pronunciation": "",
                          "naturalness": "", "voice_similarity": "", "notes": ""})
        secret_rows.append({"file": name, "variant": variant, "src": src})

    with open(os.path.join(OUT, "LISTEN.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "order", "task_id", "group"])
        w.writeheader()
        w.writerows(listen_rows)
    with open(os.path.join(OUT, "listen_form.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "text_accuracy", "pronunciation",
                                          "naturalness", "voice_similarity", "notes"])
        w.writeheader()
        w.writerows(form_rows)
    with open(os.path.join(OUT, "SECRET_map.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "variant", "src"])
        w.writeheader()
        w.writerows(secret_rows)

    by_group = {}
    for r in listen_rows:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
    readme = f"""# Слепое прослушивание s2 (варианты скрыты)

Файлы: `wav\\*.wav` ({len(items)} шт). Соответствие заданий: `LISTEN.csv` (task_id и группа видны,
вариант — НЕТ). НЕ открывайте `SECRET_map.csv` до окончания прослушивания.

Группы: {json.dumps(by_group, ensure_ascii=False)}
- phonetics: новые слова с ж/з, ч/ц, ы/и (перенос на новую лексику)
- phrase_training: одна фраза «Сбер, включи музыку для пробежек через десять минут»,
  сгенерированная как обучающие сэмплы (u10000-контроль — чистый чекпоинт).

Заполняйте `listen_form.csv` по каждому файлу (1–10):
text_accuracy (точность текста), pronunciation (произношение), naturalness (естественность),
voice_similarity (сходство голоса), notes (что услышали).

ВАЖНО: автоматический ASR (WER) измеряет только содержание и НЕ фиксирует фонетические
замены (ж/з, ч/ц, ы/и и т.п.). Слышимые дефекты имеют приоритет над высоким ASR.

После заполнения формы — открыть `SECRET_map.csv`, собрать оценки по вариантам и сверить
с авто-метриками (reports\\s2_compare\\).
"""
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)
    print(f"blind set: {len(items)} files | groups: {by_group} | {OUT}")
    print("BLIND_BUILD_DONE")


if __name__ == "__main__":
    main()
