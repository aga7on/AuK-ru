"""Judge manifest для маг-пробника s4@3000 (64 tool-семпла), судья v3 --manifest."""
import json
import os

AUK = r"G:\AI\AuK"
SRC = os.path.join(AUK, "local_tests", "s4_mag_probe_3000", "mag_probe.json", "mag_probe.json")
OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "s4_mag_judge_manifest.json")

INSTR = {
    "volume_up": "Сделай громче", "volume_down": "Сделай тише",
    "pitch_up": "Сделай голос выше", "pitch_down": "Сделай голос ниже",
    "speed_up": "Сделай речь быстрее", "speed_down": "Сделай речь медленнее",
}


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    out = []
    for e in d["rows"]:
        op = e["op"]
        mag = e["mag"]
        instr = INSTR[op]
        goal = f"{op} target={mag}"
        out.append({"id": f"s4m__{e['id']}", "mode": "tool", "group": "tool",
                    "instruction": instr, "text": e["id"], "goal": goal,
                    "checks": "направление и величина изменения, артефакты, разборчивость",
                    "ref": e["ref"], "file": e["file"], "wav": e["file"]})
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} -> {OUT}")


if __name__ == "__main__":
    main()
