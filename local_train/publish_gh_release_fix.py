# -*- coding: utf-8 -*-
"""Публикация исправленного релиза в GitHub: честные имена семплов + новый README.

Шаги:
  1. загрузить samples/sample_0{1..5}_male.wav (base64 через Contents API);
  2. удалить старые ложно именованные samples/0{1..5}_{female,male}_*.wav;
  3. заменить README.md на честную версию (README_github.md);
  4. верификация: дерево репо не содержит ложных имён, README содержит новые секции.

usage: python publish_gh_release_fix.py
"""
import base64
import json
import os
import subprocess
import sys

REPO = "aga7on/AuK-ru"
AUK = r"G:\AI\AuK"
SAMPLES = os.path.join(AUK, "local_train", "release", "samples")
README_SRC = os.path.join(AUK, "local_train", "release", "README_github.md")
OLD = ["samples/01_female_happy.wav", "samples/02_female_excited.wav",
       "samples/03_male_sad.wav", "samples/04_male_angry.wav", "samples/05_male_fearful.wav"]
NEW = [f"samples/sample_0{i}_male.wav" for i in range(1, 6)]


def gh(*a):
    r = subprocess.run(["gh"] + list(a), capture_output=True, text=True)
    return (r.returncode, r.stdout, r.stderr)


def put(path, local, msg):
    c = open(local, "rb").read()
    pl = {"message": msg, "content": base64.b64encode(c).decode(), "branch": "main"}
    rc, out, _ = gh("api", f"repos/{REPO}/contents/{path}", "--jq", ".sha")
    if rc == 0 and out.strip():
        pl["sha"] = out.strip()
    open("_p.json", "w").write(json.dumps(pl))
    rc2, o2, e2 = gh("api", "-X", "PUT", f"repos/{REPO}/contents/{path}", "--input", "_p.json")
    os.remove("_p.json")
    return rc2 == 0, (e2 or o2)[:120]


def delete(path, msg):
    rc, out, _ = gh("api", f"repos/{REPO}/contents/{path}", "--jq", ".sha")
    if rc != 0 or not out.strip():
        return True, "already absent"
    pl = {"message": msg, "sha": out.strip(), "branch": "main"}
    open("_d.json", "w").write(json.dumps(pl))
    rc2, o2, e2 = gh("api", "-X", "DELETE", f"repos/{REPO}/contents/{path}", "--input", "_d.json")
    os.remove("_d.json")
    return rc2 == 0, (e2 or o2)[:120]


def main():
    os.chdir(AUK)
    msg = ("Honesty fix: samples were all MALE voice (f0 125-182 Hz) and emotions are not "
           "acoustically separable (eta2<0.10 in 6/7 metrics, LOO acc 0.017 vs chance 0.2, "
           "p=1.0). Renamed sample_NN_male.wav, withdrawn emotion claims, added research-"
           "purpose statement and attribution/licenses under spoiler.")

    print("--- upload new samples ---")
    for i, repo_path in enumerate(NEW, start=1):
        loc = os.path.join(SAMPLES, os.path.basename(repo_path))
        if not os.path.exists(loc):
            print("SKIP (missing local):", loc)
            continue
        ok, err = put(repo_path, loc, msg)
        print(("OK   " if ok else "FAIL ") + repo_path + ("" if ok else " | " + err))

    print("--- delete false-named samples ---")
    for repo_path in OLD:
        ok, err = delete(repo_path, msg)
        print(("OK   " if ok else "FAIL ") + repo_path + ("" if ok else " | " + err))

    print("--- update README ---")
    ok, err = put("README.md", README_SRC, msg)
    print(("OK   " if ok else "FAIL ") + "README.md" + ("" if ok else " | " + err))

    # верификация
    print("--- verify tree ---")
    rc, out, err = gh("api", f"repos/{REPO}/git/trees/main?recursive=1")
    if rc == 0:
        d = json.loads(out)
        paths = [t["path"] for t in d["tree"]]
        bad = [p for p in paths if any(k in p for k in
                                       ("female_happy", "female_excited", "male_sad",
                                        "male_angry", "male_fearful"))]
        samples = sorted(p for p in paths if p.startswith("samples/") and p.endswith(".wav"))
        print("false-named left:", bad if bad else "NONE")
        print("samples in repo:", samples)
        print("total files:", len(paths))
    else:
        print("tree fetch failed:", err[:120])

    print("--- verify README content ---")
    rc, out, err = gh("api", f"repos/{REPO}/contents/README.md",
                      "-H", "Accept: application/vnd.github.raw")
    if rc == 0:
        checks = {
            "research purpose": "исследовательских и образовательных целях" in out,
            "emotion withdrawn": "заявление отозвано" in out,
            "separability table": "0.017" in out and "p = 1.0" in out,
            "samples honest names": "sample_01_male.wav" in out,
            # старые имена остаются в тексте как пояснение; ССЫЛОК на них быть не должно
            "no links to false-named files": "](samples/01_female_happy.wav)" not in out
                                             and "](samples/03_male_sad.wav)" not in out,
            "old names explained": "были **ложными**" in out,
            "attribution spoiler": "<details>" in out and "Атрибуция и лицензии" in out,
            "disclaimer": "Отказ от ответственности" in out,
            "negative results table": "НЕ дало результата" in out,
            "val_max_rows rule": "val_max_rows" in out,
            "code fence ok": "```python" in out and "\\\\python" not in out,
        }
        for k, v in checks.items():
            print(("OK   " if v else "FAIL ") + k)
        print("all passed:", all(checks.values()))
    else:
        print("readme fetch failed:", err[:120])


if __name__ == "__main__":
    main()
