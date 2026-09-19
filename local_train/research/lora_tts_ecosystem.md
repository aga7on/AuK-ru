# LoRA-TTS: экосистема Hugging Face (дайджест ресёрча, 2026-09-14)

## Выводы
- LoRA-TTS сообщества активны для: Orpheus/Llama-SNAC (самое большое), Qwen3-TTS, MOSS-TTS (LAION), Chatterbox, F5-TTS, XTTS-v2.
- **Готовых русских LoRA-TTS адаптеров НЕТ** (проверено HF + ModelScope). Русский = DIY на базах XTTS-v2 / F5-TTS_RUSSIAN / Qwen3-TTS / VoxCPM2.
- Типовой рецепт: r=16/α=32, dropout 0.05, attention q/k/v/o (+ иногда gate/up/down), LR 1e-5..1e-4, 10–50 ч или 5k–20k реплик.

## Примеры адаптеров (HF)
| Репо | База | Язык | Что адаптирует | r/α | Данные |
|---|---|---|---|---|---|
| hypaai/Hypa-Orpheus-3b-TTS-VC | Orpheus 3B | 22 языка | TTS+VC | 512/512 | 8 442 ч |
| kenpath/svara-tts-v1 | Orpheus/Indic | 19 инд. | TTS | — | 2000+ ч |
| nicolajreck/csm-1b-danish-tts | CSM-1B | датский | язык | 16/32 | 35k клипов |
| AbDhumal/orpheus-3b-turkish-tts-v2 | Orpheus | турецкий | язык | 32/64 | 20k (WER 1.58→0.72) |
| hamidfzm/MOSS-TTS-Realtime-Persian-lora | MOSS 1.7B | персидский | язык/ударения | 16/32 | 64 ч |
| instavar/f5-tts-v1-lora-female01 | F5-TTS v1 | англ. | голос | 16/16 | 10 850 |
| reenigne314/chatterbox-indic-lora | Chatterbox | 8 инд. | языки+токенайзер | 32/64 | 10–52 ч/язык |
| MAdel121/xtts-v2-egyptian-arabic-lora | XTTS-v2 | араб. егип. | диалект | 16/32 | 12 363 |
| MLA299/Tennda-Waves | XTTS-v2 | ZH/EN | диалект | 8/32 | 100 клипов/7.4 мин |
| TTS-AGI/moss-voice-profile-loras | MOSS | EN/DE | эмоции/голоса | 4/8 | 853 ч (абляция: r4≈r16) |

## Полезные ссылки
- https://huggingface.co/api/models?filter=lora&search=tts&limit=100 — живой список
- https://huggingface.co/TTS-AGI/moss-voice-profile-loras — эталон дизайна LoRA-экосистемы
- https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning
- https://github.com/odunola499/f5-lora
- https://github.com/instavar/qwen3-tts-lora-finetuning
- https://github.com/thienphucope/XTTS_V2_lora
- https://github.com/RVC-Boss/GPT-SoVITS
- Русские базы (не LoRA): Misha24-10/F5-TTS_RUSSIAN, zaakirio/kokoro-ru, Silero v5 RU
