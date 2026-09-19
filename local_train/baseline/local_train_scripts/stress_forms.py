"""Print stressed forms of the 6 problem phrases for respell A/B construction."""
import sys

sys.path.insert(0, r"G:\AI\AuK\src")
from auk.infer.infer_gradio import _accentize_ru

PHRASES = [
    "Электронное объявление появилось на сайте вчера.",
    "Пожалуйста, не опаздывайте на встречу.",
    "Привет! Это проверка русского произношения. Раз, два, три.",
    "Администратор объяснил правила пользования сервисом.",
    "Люблю грозу в начале мая, когда весенний первый гром.",
    "Сложные времена требуют простых решений.",
    "Мне нравится смотреть на звёзды летними ночами.",
]
for p in PHRASES:
    print(repr(_accentize_ru(p)))
