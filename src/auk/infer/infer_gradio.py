from __future__ import annotations

import argparse
import json
import math
import os
import re

import gradio as gr
import torch

from auk.infer.infer_auk import AukInfer, get_gen_duration
from auk.infer.pe import PromptEnhancer, PromptEnhancerError
from auk.infer.ru_translit import ru_to_latin

_ACCENTIZER = None
_SENT_SPLIT = None
_PRON_LEXICON = None


def _apply_pron_lexicon(text: str) -> str:
    """Empirically found respellings for hard words (applied after RuAccent)."""
    global _PRON_LEXICON
    if _PRON_LEXICON is None:
        import json as _json
        import re as _re

        path = os.environ.get("AUK_PRON_LEXICON", r"G:\AI\AuK\local_train\pron_lexicon.json")
        try:
            data = _json.load(open(path, encoding="utf-8"))
            _PRON_LEXICON = [(_re.compile(p["find"]), p["repl"]) for p in data.get("patterns", [])]
        except Exception:
            _PRON_LEXICON = []
    for rx, repl in _PRON_LEXICON:
        text = rx.sub(repl, text)
    return text


def _has_cyrillic(part: str) -> bool:
    return any(ch.isalpha() and ("а" <= ch.lower() <= "я" or ch in "ёЁ") for ch in part)


def _accentize_ru(text: str, use_lexicon: bool = True) -> str:
    global _ACCENTIZER, _SENT_SPLIT
    if _ACCENTIZER is None:
        from ruaccent import RUAccent

        accentizer = RUAccent()
        accentizer.load(omograph_model_size="turbo2", use_dictionary=True, device="CPU")
        _ACCENTIZER = accentizer
    if _SENT_SPLIT is None:
        import re

        _SENT_SPLIT = re.compile(r"(?<=[.!?:;\n])")
    parts = _SENT_SPLIT.split(text)
    out = []
    for part in parts:
        if _has_cyrillic(part):
            try:
                stripped = part.strip()
                core = _ACCENTIZER.process_all(stripped)
                lead = part[: len(part) - len(part.lstrip())]
                trail = part[len(part.rstrip()):]
                part = lead + core + trail
            except Exception:
                pass
        out.append(part)
    joined = "".join(out)
    return _apply_pron_lexicon(joined) if use_lexicon else joined


# Static examples derived from the public AuK demo sample list.
# Each row is: (audio path relative to the repository root, instruction, target duration seconds).
DEMO_EXAMPLE_GROUPS: dict[str, dict[str, list[tuple[str | None, str, float]]]] = {
    "Speech Generation": {
        "Zero-shot TTS": [
            (
                "assets/demo-input-audio/zero-shot-tts/ref.wav",
                "Say the following with the same voice: 'Ladies and gentlemen, it's an honor to have the opportunity to address such a distinguished audience'",
                6.0,
            ),
        ],
        "Instruct TTS": [
            (
                None,
                "Say the following in the voice described here: “一位雄才大略、性格复杂的乱世枭雄，以略显沙哑却极有穿透力的中年男声说话。语气自信、果断，带着审视人心的敏锐感。讲话时节奏变化明显，可以先压低声音缓缓铺垫，再突然加重关键字。既有豪迈，也隐约带着危险与猜疑”, and say: “宁可我负天下人，休教天下人负我。”",
                3.36,
            ),
            (
                None,
                "Say the following in the voice described here: “一位中年女性，在面对面质问多年前抛弃自己的人时，用沙哑、干涩的嗓音倾诉三十年压抑的恨意与委屈，仿佛眼泪已经流尽。语速缓慢而沉稳，字字清晰冰冷，整体音量中等偏低，但说到愤恨处声音微微发颤、压抑着哽咽，音色苍凉而有力，句尾带着渗骨的决绝，最后一句陡然拔高、几乎用尽全身力气砸出来”, and say: “哼，我的眼泪早哭干了，我没有委屈，我有的是恨，是悔，是三十年一天一天我自己受的苦。你大概已经忘了你做的事了！”",
                15.0,
            ),
            (
                None,
                "Say the following in the voice described here: “一位莎士比亚戏剧风格的经典反派，正在揭露真相前细细品味这一刻。声音是浑厚且富有戏剧张力的男中音，胸腔共鸣饱满，语速从容舒缓，带着刻意的停顿与强调；辅音被夸张地滚动、咬字清晰而富有舞台感。语调先佯装甜蜜温柔，几乎带着宠溺，随后在词句间骤然转冷，锋利如刀刃，每个字都缠绕着危险而愉悦的得意之情”, and say: “You see, my dear, the trap was never meant for you. It was always meant for him.”",
                7.0,
            ),
            (
                None,
                "Say the following in the voice described here: “仿佛在葬礼上宣布噩耗一般，声音轻柔而哽咽，强忍着泪水，话语断断续续、每句之间都有停顿，字字愈发沉重，最后一句的尾音被悲伤撕裂、颤抖着透出压抑不住的哭腔。语调缓慢低沉，气息不稳，清晰度因哽咽而略受影响，整体氛围凝重而哀恸”, and say: “He always said he'd come back. He always kept his word. Until now.”",
                7.0,
            ),
        ],
    },
    "Content Editing": {
        "Speech Content Editing": [
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.",
                7.0,
            ),
        ],
        "Lyric Editing": [
            (
                "assets/demo-input-audio/vocal-edit/vocaledit-en-1-input.wav",
                "Replace “rear view” with “like you” in the lyrics",
                5.44,
            ),
        ],
    },
    "Acoustic Editing": {
        "Pitch Editing": [
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调降低3个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调降低2个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调降低1个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调升高1个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调升高2个半音。", 5.12),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调升高3个半音。", 5.12),
        ],
        "Speed Editing": [
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为0.5倍。", 20.6),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为0.75倍。", 13.74),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为1.25倍。", 8.24),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为1.5倍。", 6.86),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为2.0倍。", 5.16),
        ],
        "Volume Editing": [
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量降低15分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量降低10分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量降低5分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量升高5分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量升高10分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量升高15分贝。", 3.78),
        ],
    },
    "Paralinguistic Editing": {
        "Emotion Editing": [
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a happy tone", 6.6),
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a sad tone", 7.6),
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a angry tone", 6.6),
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a afraid tone", 7.22),
        ],
        "Timbre Editing": [
            (
                "assets/demo-input-audio/vc/vc-1-input.wav",
                "Keep the words and change the timbre to: “这位说话人的声音低沉而浑厚，语速平稳，吐字清晰。他的说话风格沉稳而富有思考，带有平静的反思特质。”",
                6.56,
            ),
        ],
        "De-accent": [
            ("assets/demo-input-audio/accent/accent-anhui-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 4.0),
            (
                "assets/demo-input-audio/accent/accent-dongbei-input.wav",
                "请去掉这段语音里的方言口音，保持说话人音色一致。",
                17.52,
            ),
            ("assets/demo-input-audio/accent/accent-fujian-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 4.06),
            ("assets/demo-input-audio/accent/accent-hubei-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 2.2),
            ("assets/demo-input-audio/accent/accent-hunan-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 3.52),
            ("assets/demo-input-audio/accent/accent-sichuan-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 6.74),
            ("assets/demo-input-audio/accent/accent-tibetan-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 3.22),
        ],
        "Nonverbal Editing": [
            ("assets/demo-input-audio/nv/en-c-input.wav", "Add a breath before “We tested”", 10.44),
            ("assets/demo-input-audio/nv/en-c-input.wav", "Add a sneeze before “only one of them”", 12.0),
            ("assets/demo-input-audio/nv/en-c-input.wav", "Add a pause after “only one of them”", 11.0),
            ("assets/demo-input-audio/nv/en-d-input.wav", "Remove the humming", 22.0),
            ("assets/demo-input-audio/nv/en-d-input.wav", "Remove the hiss", 22.82),
            ("assets/demo-input-audio/nv/en-d-input.wav", "Remove the sobbing", 21.0),
            ("assets/demo-input-audio/nv/zh-a-input.wav", "Add a sigh before “这个月”", 13.0),
            ("assets/demo-input-audio/nv/zh-a-input.wav", "Add a filler before “主要的缺口”", 12.38),
            ("assets/demo-input-audio/nv/zh-a-input.wav", "Add a cough before “再决定”", 13.0),
            ("assets/demo-input-audio/nv/zh-b-input.wav", "Remove the laughter", 12.58),
            ("assets/demo-input-audio/nv/zh-b-input.wav", "Remove the gasp of surprise", 12.72),
            ("assets/demo-input-audio/nv/zh-b-input.wav", "Remove the sharp inhale", 12.62),
        ],
        "Whisper Conversion": [
            ("assets/demo-input-audio/whisper/wh-w2n-zh-input.wav", "Turn this into a whisper", 8.36),
        ],
    },
    "Enhancement & Separation": {
        "Speech Enhancement": [
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Remove the background noise and make the voice cleaner", 5.0),
        ],
        "Speech Separation": [
            ("assets/demo-input-audio/ss/en-1-input.wav", "Keep only the speaker who says “get what”", 28.0),
            ("assets/demo-input-audio/ss/en-1-input.wav", "Keep only the first speaker", 28.0),
            ("assets/demo-input-audio/ss/zh-1-input.wav", "Keep only the speaker who says “警队规矩”", 19.28),
            ("assets/demo-input-audio/ss/zh-1-input.wav", "Keep only the second speaker", 18.24),
        ],
        "Vocal Extraction": [
            (
                "assets/demo-input-audio/vocal-extraction/vocal-1-input.wav",
                "Extract the vocals and remove the accompaniment",
                10.88,
            ),
        ],
        "Audio Quality Enhancement": [
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Improve the audio quality and make it clearer", 5.0),
        ],
    },
}

# variant label -> checkpoint path (filled in main() from CLI args)
CKPT_PATHS: dict[str, str] = {}

# Russian-first demo set (our checkpoint; refs bundled in repo assets)
_RU_REF = "assets/demo-input-audio/ru/user_ref.wav"
DEMO_EXAMPLE_GROUPS = {
    "Speech Generation": {
        "Zero-shot TTS": [
            (_RU_REF, "Say the following in Russian with clear, natural pronunciation: 'Привет! Это проверка русского произношения.'", 4.0),
            (_RU_REF, "Say the following in Russian with clear, natural pronunciation: 'Ещё более важную роль на Африканском Роге играет устойчивое развитие.'", 6.0),
            (_RU_REF, "Say the following in Russian with clear, natural pronunciation: 'Позвони мне, пожалуйста, когда освободишься.'", 4.5),
            (_RU_REF, "Say the following in Russian with clear, natural pronunciation: 'Две тысячи двадцать шестого года мы запустили этот проект.'", 4.5),
        ],
        "Instruct TTS": [
            (None, "Say the following in a happy, energetic Russian voice: 'Скидки до пятидесяти процентов только сегодня!'", 3.5),
            (None, "Say the following in a calm, melancholic Russian voice: 'Осень ушла, и вместе с ней ушли тёплые дни.'", 5.0),
        ],
    },
    "Editing (базовые возможности)": {
        "Speech Content Editing": [
            (_RU_REF, "Replace 'привет' with 'здравствуйте'.", 10.21),
        ],
        "Emotion Editing": [
            (_RU_REF, "Say this in a happy tone", 10.21),
            (_RU_REF, "Say this in a sad tone", 10.21),
        ],
        "Whisper Conversion": [
            (_RU_REF, "Turn this into a whisper", 10.21),
        ],
    },
}


CONFIG_PATHS: dict[str, str | None] = {}
# variant label -> loaded engine, populated lazily on first use and reused after
ENGINES: dict[str, AukInfer] = {}
ENGINE_KWARGS: dict = {}
DEVICE_PATHS: dict[str, str | None] = {}

BASE_LABEL = "AuK (Base)"
FLASH_LABEL = "AuK-Flash ⚡"

SAMPLING_PRESETS = {
    BASE_LABEL: {"nfe": 64, "cfg": 2.0, "interactive": True},
    FLASH_LABEL: {"nfe": 4, "cfg": 0.0, "interactive": False},
}

LOUDNESS_LIMIT_TASK_TYPES = frozenset({"extract_vocals", "vocal_edit"})
VOCAL_OUTPUT_TARGET_LUFS = -14.0
VOCAL_OUTPUT_PEAK_CEILING = 0.95

DEMO_CSS = """
#auk-main-row {
    align-items: stretch;
}
#auk-input-column,
#auk-output-column {
    display: flex;
    flex-direction: column;
    height: 100%;
}
#auk-input-bottom,
#auk-output-bottom {
    margin-top: auto;
}
#auk-output-bottom {
    display: flex;
    flex-direction: column;
}
#auk-input-audio {
    min-height: 210px !important;
}
#auk-output-audio {
    min-height: 200px !important;
}
#auk-pe-output {
    margin-top: 12px;
}
.auk-pe-card textarea {
    min-height: 92px !important;
    height: 92px !important;
}
.auk-sampling-note {
    margin-top: -6px;
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
.auk-duration-note {
    margin-top: -8px;
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
.auk-examples-table table tbody tr > td:last-child,
.auk-examples-table table tbody tr > td:last-child * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}

/* ================= Windows Aero (green) skin ================= */
body {
    background:
        radial-gradient(1100px 520px at 50% -10%, #ffffff 0%, rgba(255,255,255,0) 60%),
        linear-gradient(180deg, #eaf7e4 0%, #cfecc4 45%, #a9d99a 100%) fixed !important;
}
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    background: transparent !important;
    font-family: "Segoe UI", "Segoe UI Variable", Tahoma, sans-serif !important;
}
.gradio-container .block,
.gradio-container .form {
    background: linear-gradient(180deg, rgba(255,255,255,0.78) 0%, rgba(244,252,240,0.55) 100%) !important;
    -webkit-backdrop-filter: blur(12px) saturate(1.2);
    backdrop-filter: blur(12px) saturate(1.2);
    border: 1px solid rgba(255,255,255,0.85) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(28,84,28,0.18), 0 1px 2px rgba(28,84,28,0.12),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
}
.auk-title {
    background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(222,243,214,0.75) 55%, rgba(196,232,182,0.70) 100%) !important;
    border: 1px solid rgba(255,255,255,0.95) !important;
    border-radius: 12px !important;
    padding: 6px 14px !important;
    box-shadow: 0 6px 18px rgba(28,84,28,0.22), inset 0 1px 0 rgba(255,255,255,1) !important;
}
.auk-title h1 {
    color: #1c5a15 !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 0 rgba(255,255,255,0.9) !important;
    letter-spacing: 0.2px;
}
.gradio-container h2, .gradio-container h3 {
    color: #205f18 !important;
    text-shadow: 0 1px 0 rgba(255,255,255,0.75);
}
button.primary, .gradio-container .primary {
    background: linear-gradient(180deg, #b7e98a 0%, #6cc23e 45%, #3f931f 55%, #357c15 100%) !important;
    border: 1px solid #2f7012 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 1px rgba(0,0,0,0.35);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.85), inset 0 -1px 0 rgba(0,0,0,0.15),
                0 2px 6px rgba(28,84,28,0.35) !important;
}
button.primary:hover, .gradio-container .primary:hover { filter: brightness(1.08) saturate(1.05); }
button.primary:active {
    filter: brightness(0.95);
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.3) !important;
}
.gradio-container button.secondary {
    background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(226,242,220,0.85)) !important;
    border: 1px solid rgba(120,170,110,0.6) !important;
    border-radius: 8px !important;
    color: #235c1b !important;
    box-shadow: inset 0 1px 0 #fff, 0 1px 3px rgba(28,84,28,0.18) !important;
}
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid rgba(130,175,120,0.55) !important;
    border-radius: 8px !important;
    box-shadow: inset 0 2px 5px rgba(28,84,28,0.12) !important;
    color: #17330f !important;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus {
    outline: none !important;
    border-color: #4f9a34 !important;
    box-shadow: 0 0 0 3px rgba(90,175,60,0.35), inset 0 2px 5px rgba(28,84,28,0.10) !important;
}
.gradio-container label > span,
.gradio-container .block-title,
.gradio-container span[data-testid="block-info"] {
    color: #2c5c23 !important;
    font-weight: 600 !important;
}
.gradio-container input[type="range"] { accent-color: #4a9a2c !important; }
.gradio-container input[type="checkbox"],
.gradio-container input[type="radio"] { accent-color: #3f8f1f !important; }
.gradio-container .tabs > .tab-nav button {
    background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(226,242,220,0.75)) !important;
    border: 1px solid rgba(140,180,130,0.55) !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    color: #2c5c23 !important;
    font-weight: 600 !important;
}
.gradio-container .tabs > .tab-nav button.selected {
    background: linear-gradient(180deg, #ffffff, #d9f0cf) !important;
    color: #174d12 !important;
    box-shadow: inset 0 1px 0 #fff !important;
}
.auk-examples-table table { border-radius: 10px !important; overflow: hidden; }
.auk-examples-table table thead { background: linear-gradient(180deg, #eaf7e4, #d5eccb) !important; }
"""

RU_TABS = {
    "Speech Generation": "Генерация речи",
    "Content Editing": "Редактирование содержания",
    "Acoustic Editing": "Акустика",
    "Paralinguistic Editing": "Паралингвистика",
    "Enhancement & Separation": "Улучшение и разделение",
    "Zero-shot TTS": "Zero-shot TTS",
    "Instruct TTS": "TTS по описанию",
    "Speech Content Editing": "Содержание речи",
    "Lyric Editing": "Текст песни",
    "Pitch Editing": "Тон",
    "Speed Editing": "Темп",
    "Volume Editing": "Громкость",
    "Emotion Editing": "Эмоция",
    "Timbre Editing": "Тембр",
    "De-accent": "Убрать акцент",
    "Nonverbal Editing": "Невербальные звуки",
    "Whisper Conversion": "Шёпот",
    "Speech Enhancement": "Очистка речи",
    "Speech Separation": "Разделение голосов",
    "Vocal Extraction": "Выделение вокала",
    "Audio Quality Enhancement": "Качество звука",
}


def ru(label: str) -> str:
    return RU_TABS.get(label, label)


def default_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, *parts)


def update_sampling_controls(variant: str):
    preset = SAMPLING_PRESETS.get(variant, SAMPLING_PRESETS[BASE_LABEL])
    interactive = preset["interactive"]
    return (
        gr.update(value=preset["nfe"], interactive=interactive),
        gr.update(value=preset["cfg"], interactive=interactive),
    )


def get_engine(variant: str) -> AukInfer:
    if variant not in ENGINES:
        ckpt_path = CKPT_PATHS.get(variant)
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise gr.Error(f"Чекпоинт {variant} не найден: {ckpt_path}")
        # config.yaml ships next to the checkpoint (release dirs bundle their own)
        ckpt_dir_config = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config.yaml")
        config_path = CONFIG_PATHS.get(variant) or ckpt_dir_config
        if not os.path.isfile(config_path):
            raise gr.Error(f"config.yaml не найден рядом с чекпоинтом {variant}: {config_path}")
        gr.Info(f"Загружаю {variant}… (первая загрузка ~15 с)")
        ENGINES[variant] = AukInfer(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device=DEVICE_PATHS.get(variant) or ENGINE_KWARGS.get("device"),
            **{key: value for key, value in ENGINE_KWARGS.items() if key != "device"},
        )
    return ENGINES[variant]


def _limit_vocal_output_loudness(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    output = waveform.to(torch.float64)
    try:
        import pyloudnorm as pyln

        measured_lufs = float(pyln.Meter(sample_rate).integrated_loudness(output.numpy()))
        if math.isfinite(measured_lufs) and measured_lufs > VOCAL_OUTPUT_TARGET_LUFS:
            gain = 10.0 ** ((VOCAL_OUTPUT_TARGET_LUFS - measured_lufs) / 20.0)
            output = output * min(1.0, gain)
    except Exception:
        pass

    if output.numel():
        peak = float(output.abs().max())
        if math.isfinite(peak) and peak > VOCAL_OUTPUT_PEAK_CEILING:
            output = output * (VOCAL_OUTPUT_PEAK_CEILING / peak)
    return output


def _format_output_audio(out_audio: torch.Tensor, sample_rate: int, task_type: str | None = None):
    waveform = out_audio.detach().to(torch.float64).cpu()
    if task_type in LOUDNESS_LIMIT_TASK_TYPES and waveform.ndim == 2:
        if waveform.shape[0] <= 8:
            waveform = waveform.mean(dim=0)
        elif waveform.shape[1] <= 8:
            waveform = waveform.mean(dim=1)
    elif waveform.ndim == 2 and 1 in waveform.shape:
        waveform = waveform.reshape(-1)
    if waveform.ndim != 1:
        raise gr.Error(f"Ожидалось моно-аудио, получена форма {tuple(waveform.shape)}.")
    if not torch.isfinite(waveform).all():
        raise gr.Error("Сгенерированное аудио содержит NaN или Inf.")
    if task_type in LOUDNESS_LIMIT_TASK_TYPES:
        waveform = _limit_vocal_output_loudness(waveform, sample_rate)
    pcm16 = torch.round(waveform.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy()
    return int(sample_rate), pcm16


def _spoken_text(text: str) -> str:
    """Extract the quoted spoken part of an instruction (fallback: whole text)."""
    import re

    matches = re.findall(r"['\"«»]([^'\"«»]{6,})['\"«»]", text)
    return max(matches, key=len) if matches else text


_TTS_TASK_PREFIX = re.compile(r"^\s*(say|speak|read|pronounce|replace|add|turn|keep|remove|extract|improve|напи|произнеси|замени|добавь|сделай)", re.I)


def _tts_wrap(text: str) -> str:
    """Wrap bare Russian text into the trained TTS template (mode 'Озвучка')."""
    import re

    if not text or not _has_cyrillic(text):
        return text
    if re.search(r"['\"«»]", text) or _TTS_TASK_PREFIX.match(text):
        return text
    return f"Say the following in Russian with clear, natural pronunciation: '{text}'"


def _limit_ref_audio(audio, max_seconds: float = 10.0):
    """Cap reference length for TTS (training refs were ~5-10s; long refs break the model)."""
    if not isinstance(audio, str) or not os.path.isfile(audio):
        return audio
    try:
        import soundfile as sf

        info = sf.info(audio)
        if info.samplerate <= 0 or info.frames / info.samplerate <= max_seconds:
            return audio
        x, sr = sf.read(audio, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        dur = len(x) / sr
        if dur > max_seconds:
            # Cut at the quietest point near the limit (phrase boundary, not mid-word).
            win = max(1, int(0.06 * sr))
            step = max(1, int(0.01 * sr))
            lo = int(6.5 * sr)
            hi = min(len(x) - win, int(min(10.5, dur) * sr))
            cut = int(max_seconds * sr)
            best_e = None
            for t0 in range(lo, hi, step):
                e = float((x[t0:t0 + win] ** 2).mean())
                if best_e is None or e < best_e:
                    best_e, cut = e, t0
            x = x[:cut]
        import tempfile
        import hashlib

        h = hashlib.md5(f"{audio}|{max_seconds}".encode()).hexdigest()[:12]
        dst = os.path.join(tempfile.gettempdir(), f"auk_ref_{h}.wav")
        if not os.path.exists(dst):
            sf.write(dst, x, sr)
        return dst
    except Exception:
        return audio


def run_generate(variant, audio, instruction, gen_seconds, ref_text, gen_text, nfe, cfg, seed, task_type=None, translit=True, accentize=True, bestofn=1, trim=True, tts_wrap=True, normalize=True):
    if not instruction:
        raise gr.Error("Укажите инструкцию.")
    if not audio and not gen_seconds and not gen_text:
        raise gr.Error("Для TTS по описанию без референсного аудио нужна целевая длительность или целевой текст.")

    instruction = instruction.strip()
    if normalize:
        # S9 Russian frontend: числа/даты/валюты/аббревиатуры → слова (rutextnorm, детерминированно)
        try:
            from auk.infer.ru_frontend import normalize_numbers
            instruction = normalize_numbers(instruction)
        except Exception:
            pass
    if tts_wrap:
        instruction = _tts_wrap(instruction)
    if accentize:
        try:
            instruction = _accentize_ru(instruction)
        except Exception:
            pass
    score_text = _spoken_text(instruction)
    if translit:
        instruction = ru_to_latin(instruction)
    if tts_wrap:
        audio = _limit_ref_audio(audio)
    content = [{"type": "text", "text": instruction}]
    if audio:
        content.append({"type": "audio", "audio": audio})
    messages = [{"role": "user", "content": content}]

    engine = get_engine(variant)
    # nfe/cfg are used as-is for base AuK; AuK-Flash ignores them (locked 4-step / CFG-off recipe).
    requested_secs = gen_seconds or 0
    gen_seconds = get_gen_duration(
        audio=(audio or None),
        ref_text=(ref_text or None),
        gen_text=(gen_text or None),
        gen_seconds=(gen_seconds or None),
    )
    try:
        g = float(gen_seconds)
    except Exception:
        g = 0.0
    if not math.isfinite(g) or g <= 0 or g > 30.0:
        g = 0.0
    if tts_wrap and requested_secs <= 0:
        # Long duration windows (e.g. ref-length fallback) make the model echo the
        # reference instead of speaking the text; estimate from the spoken text.
        spoken = _spoken_text(instruction)
        if spoken and _has_cyrillic(spoken):
            g = max(3.5, min(12.0, 1.0 + 0.075 * len(spoken)))
    if g <= 0:
        g = 6.0
    gen_seconds = min(g, 30.0)
    n_cand = max(1, int(bestofn or 1))
    if n_cand > 1:
        base_seed = int(seed) if seed is not None else 1234
        seeds = [base_seed + i for i in range(n_cand)]
    else:
        seeds = [int(seed) if seed is not None else None]

    from auk.infer import quality

    def _gen(instr_text, secs, s):
        content_i = [{"type": "text", "text": instr_text}]
        if audio:
            content_i.append({"type": "audio", "audio": audio})
        return engine.generate([{"role": "user", "content": content_i}],
                               audio=(audio or None), gen_seconds=secs, nfe=int(nfe),
                               cfg_strength=float(cfg), seed=s)

    plan = [(instruction, gen_seconds, s, "base") for s in seeds]
    body = _spoken_text(instruction)
    if tts_wrap and body and audio:
        tight = max(2.4, gen_seconds - 0.8)
        t3 = f"Pronounce the following in Russian: '{body}'"
        plan.append((t3, tight, seeds[0], "t3+tight"))
        plan.append((instruction, tight, seeds[0], "tight"))
        plan.append((t3, gen_seconds, seeds[0], "template3"))

    candidates = []
    scored = []
    allowed = len(plan[:5])
    for instr_text, secs, s, tag in plan[:5]:
        out_audio, sr = _gen(instr_text, secs, s)
        idx = len(candidates)
        candidates.append((out_audio, sr))
        heard = ""
        try:
            x = quality._to_mono_np(out_audio)
            heard = quality.transcribe(out_audio, sr)
            wer = quality.wer_metrics(score_text, heard)["wer"]
            clip = quality.clip_ratio(x)
            excess = quality.pause_excess(x, sr)
        except Exception:
            wer, clip, excess = 1.0, 1.0, 0.0
        scored.append((idx, round(wer, 2), clip > 5e-4, clip, excess))
        if idx == 0:
            if len(heard.split()) < 2:
                # The reference itself is unusable (no speech recognized) — retries won't help.
                break
            if wer <= 0.5:
                # Already readable speech: one repair attempt is enough (~90s max).
                allowed = min(allowed, 2)
        if wer <= 0.15 and len(candidates) >= len(seeds):
            break
        if len(candidates) >= allowed:
            break
    chosen = min(scored, key=lambda t: (t[1], t[2], t[3], t[4], t[0]))[0]
    out_audio, sr = candidates[chosen]
    if trim:
        out_audio = quality.trim_silence(out_audio, sr)
    out_audio = quality.normalize_rms(out_audio)
    out_audio = quality.limit_peak(out_audio)
    return _format_output_audio(out_audio, sr, task_type=task_type)


def run_generate_with_pe(use_pe, variant, audio, instruction, gen_seconds, ref_text, gen_text, nfe, cfg, seed, translit=True, accentize=True, bestofn=False, trim=True, tts_wrap=True):
    n_cand = 2 if bestofn else 1
    tts_wrap = str(tts_wrap).strip().lower().startswith("озвучка")
    if not use_pe:
        if not gen_seconds and not (instruction and _has_cyrillic(instruction)):
            raise gr.Error("Укажите длительность больше 0 или включите Prompt Enhancer.")
        return run_generate(variant, audio, instruction, gen_seconds, ref_text, gen_text, nfe, cfg, seed, translit=translit, accentize=accentize, bestofn=n_cand, trim=trim, tts_wrap=tts_wrap), None

    prepared = None
    try:
        requested_duration = float(gen_seconds or 0)
        prepared = PromptEnhancer().prepare(
            instruction,
            audio,
            target_duration=requested_duration if requested_duration > 0 else None,
        )
        effective_duration = requested_duration if requested_duration > 0 else prepared.gen_seconds
        try:
            ed = float(effective_duration)
        except Exception:
            ed = 0.0
        if not math.isfinite(ed) or ed <= 0 or ed > 30.0:
            ed = 0.0
        effective_duration = ed
        generated = run_generate(
            variant,
            prepared.audio,
            prepared.instruction,
            effective_duration,
            prepared.ref_text,
            prepared.gen_text,
            nfe,
            cfg,
            seed,
            prepared.task_type,
            translit=translit,
            accentize=accentize,
            bestofn=n_cand,
            trim=trim,
            tts_wrap=tts_wrap,
        )
        asr = getattr(prepared, "asr", None)
        asr_text = asr.text if asr and asr.text else None
        task = str(getattr(prepared, "task_type", ""))
        operation_subtype = getattr(prepared, "operation_subtype", None)
        if operation_subtype:
            task = f"{task} / {operation_subtype}"
        metadata = {
            "task": task,
            "duration": f"{effective_duration:.2f} s",
            "asr_content": asr_text or "",
            "instruction": prepared.instruction,
        }
        return generated, metadata
    except (PromptEnhancerError, ValueError, FileNotFoundError) as exc:
        raise gr.Error(f"Сбой Prompt Enhancer: {type(exc).__name__}: {exc}") from None
    finally:
        if prepared is not None:
            prepared.cleanup()


def update_pe_outputs(metadata: dict | None):
    if not metadata:
        return "", "", ""
    return (
        f"Задача: {metadata.get('task') or ''}\nЦелевая длительность: {metadata.get('duration') or ''}",
        str(metadata.get("asr_content") or ""),
        str(metadata.get("instruction") or ""),
    )


def load_demo_example(index: int | None, examples: list[tuple[str | None, str, float]]):
    if index is None or not 0 <= int(index) < len(examples):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    audio, instruction, duration = examples[int(index)]
    return audio, instruction, duration, False, 1234


def _resolve_demo_examples(examples: list[tuple[str | None, str, float]]) -> list[tuple[str | None, str, float]]:
    resolved = []
    for audio, instruction, duration in examples:
        audio_path = default_path(*audio.split("/")) if audio else None
        if audio_path is None or os.path.isfile(audio_path):
            resolved.append((audio_path, instruction, duration))
    return resolved


def _demo_dataset_samples(task: str, examples: list[tuple[str | None, str, float]]) -> list[list[str | None]]:
    if task == "Instruct TTS":
        return [[instruction] for _, instruction, _ in examples]
    return [[audio, instruction] for audio, instruction, _ in examples]


def build_demo() -> gr.Blocks:
    available_variants = [label for label, path in CKPT_PATHS.items() if path and os.path.isfile(path)]
    if not available_variants:
        raise RuntimeError("No valid model checkpoint was configured.")
    initial_variant = available_variants[0]
    initial_sampling = SAMPLING_PRESETS.get(initial_variant, SAMPLING_PRESETS[BASE_LABEL])
    with gr.Blocks(title="AuK — генерация и редактирование речи") as demo:
        gr.Markdown(
            '<div align="center">\n\n# AuK — генерация и редактирование речи\n\n</div>',
            elem_classes="auk-title",
        )
        gr.Markdown(
            "**Поддерживаемые задачи**\n"
            "- **Генерация речи**: Zero-shot TTS · TTS по описанию голоса\n"
            "- **Редактирование содержания**: правка текста песни · правка содержания речи\n"
            "- **Акустическое редактирование**: тон · темп · громкость\n"
            "- **Паралингвистика**: эмоция · тембр · акцент · невербальные звуки · шёпот\n"
            "- **Улучшение и разделение**: очистка речи · разделение голосов · выделение вокала\n\n"
            "Приоритет длительности: заданная длительность → оценка по тексту → длина исходного аудио."
        )
        with gr.Row(equal_height=True, elem_id="auk-main-row"):
            with gr.Column(elem_id="auk-input-column"):  # left: inputs (audio + text)
                in_audio = gr.Audio(
                    label="Входное аудио (необязательно; пусто — для TTS по описанию)",
                    type="filepath",
                    elem_id="auk-input-audio",
                )
                in_instr = gr.Textbox(label="Инструкция", lines=4)
                with gr.Column(elem_id="auk-input-bottom"):
                    in_mode = gr.Radio(
                        ["Озвучка (TTS)", "Редактирование"],
                        value="Озвучка (TTS)",
                        label="Режим",
                        info="Озвучка: чистый русский текст сам оборачивается в обученный шаблон. Редактирование: инструкция передаётся как есть (Replace/Emotion/Whisper и др. базовые задачи).",
                    )
                    use_pe = gr.Checkbox(
                        value=False,
                        label="Улучшение запроса (Prompt Enhancer)",
                        info="PE сам определяет задачу и инструкцию; длительность оценивается, только если она равна 0.",
                    )
                    in_translit = gr.Checkbox(
                        value=False,
                        label="Латинизация русского",
                        info="Кириллица → латиница. На кириллице с ударениями произношение чётче (особенно мягкий знак) — включайте только при проблемах.",
                    )
                    in_accents = gr.Checkbox(
                        value=True,
                        label="Автоударения (RuAccent)",
                        info="Автоматически расставляет ударения в русских словах перед синтезом (включая омографы: зáмок/замóк).",
                    )
                    in_bestofn = gr.Checkbox(
                        value=False,
                        label="Авто-выбор лучшего (2 варианта, медленнее)",
                        info="Генерирует два варианта (соседние сиды) и выбирает лучший: GigaAM-распознавание текста + минимум лишних пауз.",
                    )
                    in_trim = gr.Checkbox(
                        value=True,
                        label="Сжимать лишние паузы",
                        info="Убирает долгую тишину в начале/конце и слишком длинные паузы внутри.",
                    )
                    # Keep empty internal states for the existing generation callback
                    # after removing the two manual text boxes from the UI.
                    in_ref_text = gr.State(value="")
                    in_gen_text = gr.State(value="")
            with gr.Column(elem_id="auk-output-column"):  # right: inference settings + output
                in_variant = gr.Radio(
                    choices=available_variants,
                    value=initial_variant,
                    label="Модель",
                )
                in_secs = gr.Slider(
                    0,
                    30,
                    value=0,
                    step=0.5,
                    label="Длительность (сек)",
                )
                gr.Markdown(
                    "0 = автоподбор длительности (нужен PE); значение больше 0 переопределяет длительность PE.",
                    elem_classes="auk-duration-note",
                )
                with gr.Row():
                    in_nfe = gr.Slider(
                        4,
                        64,
                        value=initial_sampling["nfe"],
                        step=1,
                        label="Шагов NFE",
                        interactive=initial_sampling["interactive"],
                        scale=2,
                    )
                    in_cfg = gr.Slider(
                        0.0,
                        5.0,
                        value=initial_sampling["cfg"],
                        step=0.1,
                        label="Сила CFG",
                        interactive=initial_sampling["interactive"],
                        scale=2,
                    )
                    in_seed = gr.Number(label="Сид (необязательно)", value=1234, precision=0, scale=1)
                gr.Markdown(
                    "Base: NFE 32 / CFG 2.0 · Flash: NFE 4 / CFG 0 (фиксировано)",
                    elem_classes="auk-sampling-note",
                )
                gen_btn = gr.Button("Сгенерировать", variant="primary")
                with gr.Column(elem_id="auk-output-bottom"):
                    out_audio = gr.Audio(label="Результат", elem_id="auk-output-audio")
                    pe_metadata_state = gr.State(value=None)

        with gr.Column(elem_id="auk-pe-output"):
            gr.Markdown("### Результат Prompt Enhancer")
            with gr.Row(equal_height=True):
                pe_summary = gr.Textbox(
                    label="Сводка",
                    interactive=False,
                    lines=3,
                    max_lines=3,
                    elem_classes="auk-pe-card",
                    scale=1,
                )
                pe_asr_content = gr.Textbox(
                    label="Распознанный текст (ASR)",
                    interactive=False,
                    lines=3,
                    max_lines=3,
                    elem_classes="auk-pe-card",
                    scale=2,
                )
                pe_instruction = gr.Textbox(
                    label="Инструкция",
                    interactive=False,
                    lines=3,
                    max_lines=3,
                    elem_classes="auk-pe-card",
                    scale=2,
                )

        in_variant.change(
            update_sampling_controls,
            inputs=in_variant,
            outputs=[in_nfe, in_cfg],
            queue=False,
            api_visibility="private",
        )
        generate_event = gen_btn.click(
            run_generate_with_pe,
            inputs=[use_pe, in_variant, in_audio, in_instr, in_secs, in_ref_text, in_gen_text, in_nfe, in_cfg, in_seed, in_translit, in_accents, in_bestofn, in_trim, in_mode],
            outputs=[out_audio, pe_metadata_state],
            api_name="run_generate_with_pe",
        )
        generate_event.then(
            update_pe_outputs,
            inputs=pe_metadata_state,
            outputs=[pe_summary, pe_asr_content, pe_instruction],
            queue=False,
            api_visibility="private",
        )

        gr.Markdown(
            "## Примеры\nВыберите пример — он подставится в панель, затем нажмите **Сгенерировать**. Можно менять сид."
        )
        with gr.Tabs():
            for category, tasks in DEMO_EXAMPLE_GROUPS.items():
                with gr.Tab(ru(category)):
                    with gr.Tabs():
                        for task, task_examples in tasks.items():
                            examples = _resolve_demo_examples(task_examples)
                            if not examples:
                                continue
                            with gr.Tab(ru(task)):
                                is_instruct_tts = task == "Instruct TTS"
                                components = [gr.Textbox(label="Инструкция", render=False)]
                                headers = ["Инструкция"]
                                if not is_instruct_tts:
                                    components.insert(0, gr.Audio(label="Аудио", render=False))
                                    headers.insert(0, "Аудио")
                                dataset = gr.Dataset(
                                    components=components,
                                    samples=_demo_dataset_samples(task, examples),
                                    headers=headers,
                                    type="index",
                                    layout="table",
                                    samples_per_page=8,
                                    show_label=False,
                                    elem_classes="auk-examples-table",
                                )
                                dataset.select(
                                    lambda index, rows=examples: load_demo_example(index, rows),
                                    inputs=dataset,
                                    outputs=[in_audio, in_instr, in_secs, use_pe, in_seed],
                                    queue=False,
                                    api_visibility="private",
                                    show_progress="hidden",
                                )

    return demo


def main():
    p = argparse.ArgumentParser(description="AuK Gradio demo")
    p.add_argument(
        "--base_ckpt",
        default=None,
        help="AuK checkpoint. If any checkpoint flag is supplied, only explicitly supplied variants are shown.",
    )
    p.add_argument(
        "--flash_ckpt",
        default=None,
        help="AuK-Flash checkpoint. If any checkpoint flag is supplied, only explicitly supplied variants are shown.",
    )
    p.add_argument("--base_config", default=None)
    p.add_argument("--flash_config", default=None)
    p.add_argument("--base_device", default=None, help="Device for AuK, for example cuda:0.")
    p.add_argument("--flash_device", default=None, help="Device for AuK-Flash, for example cuda:1.")
    p.add_argument("--preload", action="store_true")
    p.add_argument("--qwen_path", default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--cpu_offload",
        action="store_true",
        help="offload the Qwen text encoder and DiT to CPU when idle; keep the VAE on CUDA",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    explicitly_configured = args.base_ckpt is not None or args.flash_ckpt is not None
    candidates = {
        BASE_LABEL: (
            args.base_ckpt if explicitly_configured else default_path("ckpts", "AuK", "auk_base.safetensors"),
            args.base_config,
        ),
        FLASH_LABEL: (
            args.flash_ckpt if explicitly_configured else default_path("ckpts", "AuK-Flash", "auk_flash.safetensors"),
            args.flash_config,
        ),
    }
    for label, (checkpoint, config) in candidates.items():
        if checkpoint is None:
            continue
        if not os.path.isfile(checkpoint):
            if explicitly_configured:
                p.error(f"{label} checkpoint not found: {checkpoint}")
            continue
        if config is not None and not os.path.isfile(config):
            p.error(f"{label} config not found: {config}")
        CKPT_PATHS[label] = checkpoint
        CONFIG_PATHS[label] = config

    ENGINE_KWARGS.update(device=args.device, dtype=args.dtype, qwen_path=args.qwen_path, cpu_offload=args.cpu_offload)
    DEVICE_PATHS[BASE_LABEL] = args.base_device
    DEVICE_PATHS[FLASH_LABEL] = args.flash_device

    if not CKPT_PATHS:
        p.error("No valid --base_ckpt or --flash_ckpt was found.")
    if args.preload:
        for variant in CKPT_PATHS:
            get_engine(variant)

    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.slate),
        css=DEMO_CSS,
    )


if __name__ == "__main__":
    main()
