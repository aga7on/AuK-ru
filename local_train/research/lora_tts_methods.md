# LoRA для TTS: методы и практики (дайджест ресёрча, 2026-09-14)

## Наш класс моделей = F5-TTS / VoxCPM2 / CosyVoice3 (flow-matching DiT, замороженные энкодер+VAE)

### Таргеты
- Везде базово **attention q/k/v/o** (F5, VoxCPM2, GPT-SoVITS CFM, VoiceTailor, Qwen3).
- FFN (gate/up/down) — только у LLM-TTS (Orpheus/Unsloth); для акцента не доказан. VoxCPM2 заморозил FFN — всё равно +0.38 MOS.
- **НЕ трогать текстовый LLM-энкодер** (CosyVoice ablation: LLM-LoRA хуже DiT-LoRA на малых данных; Fish: медленный трансформер деградирует).
- Наша конфигурация (attn+FF+fusion+AdaLN) — надмножество доказанного; риска нет, откатимся на attention-only при перетрене.

### Гиперпараметры (рекомендации)
- **Rank 16–32** для акцента/языка; 64 при большой смене распределения (VoxCPM2: r64 лучший MOS); r128 хуже r64.
- **α = 2r** (совпадает с большинством: F5, VoxCPM2, Orpheus, Qwen3; CosyVoice3 = 4r, Fish = 0.5r).
- Dropout 0–0.1 (0.05 типовой; 0 для маленьких данных).
- **LR 5e-5…1e-4** для DiT-LoRA (F5 1e-4, VoiceTailor 1e-4, CosyVoice3 5e-5, Accent Vector 3e-5). Для AR-LLM TTS ниже (2e-6), но это не наш случай.
- Сходимость быстрая (F5: 1 эпоха/1250 шагов; Fish fast-only ~500; VoiceTailor 500 итераций), но это для узких задач; для смены акцента — больше.
- **Выбор чекпоинта — только perceptual** (F5: loss плоский; VoxCPM2: loss≠MOS). У нас уже так (судья+watcher).

### Данные
- Акцент/язык: **4–26 ч** (Zonos-Hebrew 4 ч; VoxCPM2 26 ч; Fish 55 ч; Kokoro 51 ч), **мультиспикерно** (иначе дрейф к одному голосу — StyleTTS2).
- Голос-стадия: 30–60 мин целевого голоса (Qwen3 10–30 мин, IndexTTS/XTTS/F5 ≥30 мин).
- 24 кГц, +1 с тишины в конце, без тегов (Qwen3-практики).

### Пять ключевых уроков для нас
1. Attention-first, энкодер не трогаем — совпадает с нашим дизайном.
2. Раскатка ранга: r16→r32→r64 (сейчас r32), r128 не брать.
3. Replay против забывания: 0.5–5% батча или ~10 сэмплов/спикер — сильнее model merging и EWC.
4. **LoRA-scale на инференсе** — регулируемый «диалект-кран» (Qwen3 0.3–0.35; Accent Vector: сила акцента масштабируема; ценой WER).
5. Метрики акцента: WER/UTMOS нечувствительны к акценту (ISCA 2025). Надёжнее: accent-ID, форманты F1/F2 RMSE, DTW-PPG vs нативный эталон. Наш JSON-судья с `accent` — субъективный, добавить форманты — следующий шаг.

### Ссылки
- https://arxiv.org/html/2603.07534 (Accent Vector, XTTS LoRA r16, акцент-контроль)
- https://arxiv.org/html/2606.26618 (VoxCPM2 low-resource, r64, MOS+0.38)
- https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf (PEFT-TTS, F5-based)
- https://github.com/instavar/f5-tts-lora-finetuning / https://github.com/leyoisaboy/... (CosyVoice3 LoRA)
- https://github.com/leeoisaboy/lora-cosyvoice123-chanting (DiT-LoRA > LLM-LoRA)
- https://arxiv.org/html/2408.14739v2 (VoiceTailor: attention-модули под LoRA)
- https://arxiv.org/html/2505.17496 (анти-забывание, replay)
- https://www.isca-archive.org/interspeech_2025/zhong25c_interspeech.pdf (метрики акцента)
