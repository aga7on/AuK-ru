# Пересечение tools_v3 (v3) и s2-миксов (16.09.2026)

Источник: `tools_overlap.py`; метаданные — `corpus/split_manifest.jsonl` (cluster/split).
Старый `AUDIT_2026-09-16_tools.json` относился к tools v2 и не может ничего утверждать про v3.

| проверка | результат |
|---|---|
| tools_train_ids | 6400 |
| tools_val_ids | 800 |
| tools_train_vs_val_ids | 0 |
| tools_train_vs_val_clusters | 0 |
| tools_val_vs_s2_train_ids | 0 |
| tools_train_vs_s2_train_ids | 0 |
| tools_val_vs_s2_val_ids | 0 |
| s2_train_ids | 24634 |
| s2_val_ids | 87 |
| tools_train_speaker_clusters | 3200 |
| tools_val_speaker_clusters | 400 |
| s2_train_speaker_clusters | 8360 |
| tools_train_vs_s2_train_clusters | 2189: 5, 6, 11, 17, 20, 28, 29, 31, 32, 33 |
| tools_val_vs_s2_train_clusters | 317: 9, 18, 23, 38, 58, 63, 69, 81, 92, 93 |
| tools_val_vs_s2_val_clusters | 0 |

## Соответствие source_split метаданным корпуса

| source_split (meta) -> split (corpus) | строк |
|---|---:|
| train -> train | 18207 |
| train -> None | 3200 |
| val -> train | 2242 |
| val -> None | 400 |

Несовпадений cluster meta vs corpus (val): 0

## Content-hash общих id (до 20)

| id | sha256 (первые 16) | corpus split | corpus cluster |
|---|---|---|---|

## Вывод

- tools train/val по source id пересекаются: **0**; по speaker cluster: **0**.
- tools val source id, встречающиеся в s2-речевом train: **0**.
- tools train source id, встречающиеся в s2-речевом train: **0**.
- **Кластерное пересечение**: tools train ∩ s2 train clusters = **2189**; tools val ∩ s2 train clusters = **317**; tools val ∩ s2 val clusters = **0**.
- **Расхождение сплитов**: 2242 val-строки tools имеют `source_split=val`, но их source clip id в `split_manifest.jsonl` помечен как `train` (400 val-строк — шумовые `noise_in_val_*`, вне манифеста). То есть заявленный val набор tools_v3 не является held-out относительно разметки корпуса.
- Прямого совпадения source id между tools и s2-речевым train нет (0), но 317 speaker-кластеров пересекаются → оценка инструментов не является speaker-disjoint.
- Вывод формируется по фактическим id/cluster/split, а не по прежнему v2-отчёту; build_report.json v3 сам сообщает clip_overlap_train_val=0 по СВОЕМУ сплиту, что расходится с corpus split.
