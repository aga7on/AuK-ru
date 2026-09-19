"""Append a completed eval dir to the controller history (score via the standard composite)."""
import json
import os
import sys

LOCAL = os.path.join(r"G:\AI\AuK", "local_train")
sys.path.insert(0, LOCAL)

from auto_controller import judge_means, score_results  # noqa: E402


def main():
    gen_dir, update, run_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    results_path = os.path.join(gen_dir, "results.json")
    results = json.load(open(results_path, encoding="utf-8"))
    score, per = score_results(results)
    means = judge_means(results)
    row = {"ts": "", "label": "lora", "update": update,
           "score": round(score, 4), "per_prompt": per, "means": means, "n_prompts": len(results),
           "results": results_path, "manual": True}
    import datetime
    row["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    control = os.path.join(run_dir, "control.jsonl")
    with open(control, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("appended row u%s score=%.4f overall=%.1f accent=%.1f palat=%.1f nat=%.1f sim=%.1f" % (
        update, score, means["overall"], means["accent"], means["palatalization"],
        means["naturalness"], means["voice_similarity"]))


if __name__ == "__main__":
    main()
