"""Qwen2.5-Omni judge: hears REFERENCE + SAMPLE, scores Russian pronunciation/voice match; JSON out.

Also computes objective metrics (DNSMOS OVRL, median F0, duration ratio) per sample.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ru_metrics import text_metrics  # noqa: E402

AUK_ROOT = r"G:\AI\AuK"
QWEN_PATH = os.path.join(AUK_ROOT, "ckpts", "Qwen2.5-Omni-3B")

TEMPLATE = """You are a strict expert evaluator of synthesized Russian speech. You will hear TWO audio clips:
1) REFERENCE: the target speaker's voice (ground truth recording of the same sentence).
2) SAMPLE: synthesized speech that should say the intended text and sound like the reference speaker with natural Russian pronunciation.

The intended text (the mark + stands before the stressed vowel; apostrophe ' marks softness; ignore the marks while reading):
"{text}"

Score each criterion with an integer 0-10 using these anchors:
- text_fidelity: all words and syllables present. 10 = complete; 5 = one word dropped/mangled; 0 = meaning lost. A dropped syllable inside a word (e.g. "развитие" -> "разитие") is 3 or lower.
- endings: words pronounced TO THE END; 10 = every final consonant/syllable clearly audible; 5 = some endings smeared; 0 = words swallowed, endings missing.
- naturalness: liveliness, not robotic. 10 = indistinguishable from a human announcer; 5 = audibly synthetic/monotone; 0 = mechanical voice.
- prosody: rhythm and intonation. 10 = smooth natural melody, logical pauses, steady tempo; 5 = choppy, artificial or misplaced pauses; 0 = chaotic.
- accent: 10 = native; 7 = slight accent; 3 = obvious foreign; 0 = heavy American accent.
- palatalization: soft/hard consonants (день, семья, объём, съел). 10 = all correct; 5 = some softnesses lost; 0 = everything hard.
- stress: word stress. 10 = all correct; list wrong ones in words_misstressed.
- artifacts: cleanliness. 10 = no noise; 5 = sandy/hissy artifacts; 0 = crackle, rumble, cut-offs.
- voice_similarity: how closely the SAMPLE matches the REFERENCE voice (timbre, pitch, gender, character).

Flags:
- same_gender: is the SAMPLE voice the same gender as the REFERENCE?
- text_matches: does the SAMPLE say the intended text (minor slips allowed)?
- truncated: any word cut off / unfinished?
- native_ok: would the SAMPLE pass as native Russian pronunciation?
- words_misstressed: list of words with wrong stress.
- issues: the main problems in one or two short sentences.

Respond with ONLY a JSON object (no markdown, no text around it), exactly these keys:
{{"text_fidelity": int, "endings": int, "naturalness": int, "prosody": int, "accent": int, "palatalization": int, "stress": int, "artifacts": int, "voice_similarity": int, "same_gender": true/false, "text_matches": true/false, "truncated": true/false, "native_ok": true/false, "words_misstressed": [], "issues": "string", "overall": int, "verdict": "годен|доработка|брак"}}"""


def f0_median(x, sr):
    frame = int(0.04 * sr)
    hop = int(0.02 * sr)
    if len(x) < frame * 2:
        return 0.0
    n_frames = 1 + (len(x) - frame) // hop
    win = np.hanning(frame).astype(np.float32)
    min_lag = int(sr / 400)
    max_lag = int(sr / 60)
    f0s = []
    for i in range(n_frames):
        seg = x[i * hop: i * hop + frame] * win
        if np.sqrt(np.mean(seg * seg)) < 1e-3:
            continue
        seg = seg - seg.mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(seg, 2 * frame)) ** 2)[:max_lag + 1]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        band = ac[min_lag:max_lag + 1]
        if len(band) == 0 or float(band.max()) < 0.35:
            continue
        f0s.append(sr / (min_lag + int(np.argmax(band))))
    return float(np.median(f0s)) if f0s else 0.0


def read_mono(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def dnsmos_ovrl(x, sr, dnsmos_mod):
    wav16 = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 16000 / sr)),
                      np.arange(len(x)), x).astype(np.float32)
    d = dnsmos_mod.run(wav16, 16000)
    return float(d.get("ovrl_mos", 0.0))


ASR_PROMPT = "Transcribe this audio verbatim in Russian. Return only the transcription text, nothing else."


_NUM_WORDS = {
    "ноль", "один", "одна", "одно", "одного", "две", "два", "двух", "три", "трех", "четыре",
    "четырех", "пять", "пяти", "шесть", "шести", "семь", "семи", "восемь", "восьми", "девять",
    "девяти", "десять", "десяти", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать", "двадцать",
    "двадцати", "тридцать", "тридцати", "сорок", "сорока", "пятьдесят", "пятидесяти",
    "шестьдесят", "шестидесяти", "семьдесят", "семидесяти", "восемьдесят", "восьмидесяти",
    "девяносто", "девяноста", "сто", "ста", "двести", "двухсот", "триста", "трехсот",
    "четыреста", "четырехсот", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот",
    "тысяча", "тысячи", "тысяч", "миллион", "миллиона", "миллионов",
    "первого", "второго", "третьего", "четвертого", "пятого", "шестого", "седьмого",
    "восьмого", "девятого", "десятого", "двадцатого", "тридцатого",
}


def _norm_words(text):
    text = text.lower().replace("ё", "е").replace("+", "").replace("'", "")
    text = re.split(r"\b(human|user|assistant)\b", text)[0]
    text = text.split("\n")[0]
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    toks = []
    for w in text.split():
        if re.fullmatch(r"[0-9]+", w) or w in _NUM_WORDS:
            toks.append("<NUM>")
        elif re.fullmatch(r"[a-z0-9]+", w):
            continue
        else:
            toks.append(w)
    out = []
    for t in toks:
        if t == "<NUM>" and out and out[-1] == "<NUM>":
            continue
        out.append(t)
    return out


def recall_and_missing(ref_text, hyp_text):
    from collections import Counter

    rw, hw = _norm_words(ref_text), _norm_words(hyp_text)
    rc, hc = Counter(rw), Counter(hw)
    hits = sum((rc & hc).values())
    recall = hits / max(len(rw), 1)
    missing = sorted((rc - hc).elements())
    return recall, missing


def wer(ref_text, hyp_text):
    recall, _ = recall_and_missing(ref_text, hyp_text)
    return 1.0 - recall


def parse_json(text):
    m = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for cand in reversed(m):
        try:
            obj = json.loads(cand)
            if "overall" in obj:
                return obj
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    from speechmos import dnsmos as dnsmos_mod

    manifest = json.load(open(os.path.join(args.gen_dir, "manifest.json"), encoding="utf-8"))
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_PATH)

    results = []
    for item in manifest:
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": item["ref"]},
            {"type": "audio", "audio": item["gen"]},
            {"type": "text", "text": TEMPLATE.format(text=item["gen_text"] if "gen_text" in item else item["text"])},
        ]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(text=prompt, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=192)
        reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        judge = parse_json(reply)

        # ASR pass: target audio only -> transcription -> WER against intended text
        asr_conv = [{"role": "user", "content": [
            {"type": "audio", "audio": item["gen"]},
            {"type": "text", "text": ASR_PROMPT},
        ]}]
        asr_prompt = processor.apply_chat_template(asr_conv, add_generation_prompt=True, tokenize=False)
        a_audios, a_images, a_videos = process_mm_info(asr_conv, use_audio_in_video=False)
        a_inputs = processor(text=asr_prompt, audio=a_audios, images=a_images, videos=a_videos,
                             return_tensors="pt", padding=True, use_audio_in_video=False)
        a_inputs = a_inputs.to(model.device)
        with torch.no_grad():
            a_out = model.generate(**a_inputs, use_audio_in_video=False, max_new_tokens=128)
        asr_text = processor.batch_decode(a_out[:, a_inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        tm = text_metrics(item["text"], asr_text)
        if tm["recall"] < 0.6:
            # one retry: ASR sometimes hallucinates or truncates on the first pass
            with torch.no_grad():
                a_out2 = model.generate(**a_inputs, use_audio_in_video=False, max_new_tokens=96)
            asr2 = processor.batch_decode(a_out2[:, a_inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            tm2 = text_metrics(item["text"], asr2)
            if tm2["recall"] > tm["recall"]:
                asr_text, tm = asr2, tm2
        w = tm["wer"]

        gx, gsr = read_mono(item["gen"])
        rx, rsr = read_mono(item["ref"])
        clip = float(np.mean(np.abs(gx) >= 0.985)) if gx.size else 0.0
        obj = {
            "gen_dur": round(len(gx) / gsr, 2),
            "ref_dur": round(len(rx) / rsr, 2),
            "dur_ratio": round((len(gx) / gsr) / max(len(rx) / rsr, 0.01), 2),
            "gen_f0": round(f0_median(gx, gsr), 1),
            "ref_f0": round(f0_median(rx, rsr), 1),
            "gen_dnsmos": round(dnsmos_ovrl(gx, gsr, dnsmos_mod), 2),
            "gen_clip": round(clip, 5),
        }
        results.append({"idx": item["idx"], "text": item["text"], "gen": item["gen"], "ref": item["ref"],
                        "judge": judge, "objective": obj, "asr": asr_text, "wer": round(w, 3),
                        "cer": round(tm["cer"], 3), "recall": round(tm["recall"], 3),
                        "missing_words": tm["deletions"], "substitutions": tm["substitutions"],
                        "insertions": tm["insertions"], "adjacent_dupes": tm["adjacent_dupes"],
                        "raw_reply": reply[:500]})
        jp = (judge or {})
        print(f"[{item['idx']}] overall={jp.get('overall')} verdict={jp.get('verdict')} "
              f"text={jp.get('text_fidelity')} endings={jp.get('endings')} nat={jp.get('naturalness')} "
              f"prosody={jp.get('prosody')} palat={jp.get('palatalization')} stress={jp.get('stress')} "
              f"artifacts={jp.get('artifacts')} wer={tm['wer']:.2f} cer={tm['cer']:.2f} "
              f"ins={tm['insertions']} clip={clip:.4f} | {asr_text[:50]}",
              flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
