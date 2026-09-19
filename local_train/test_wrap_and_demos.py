"""Verify demo groups + _tts_wrap behavior."""
import sys

sys.path.insert(0, r"G:\AI\AuK\src")
from auk.infer.infer_gradio import DEMO_EXAMPLE_GROUPS, _tts_wrap  # noqa: E402

print("groups:", list(DEMO_EXAMPLE_GROUPS.keys()))
for cat, tasks in DEMO_EXAMPLE_GROUPS.items():
    for task, ex in tasks.items():
        print(" ", cat, "->", task, len(ex), "examples")

t1 = _tts_wrap("Привет! Это проверка русского произношения.")
t2 = _tts_wrap("Say the following in Russian with clear, natural pronunciation: 'тест'")
t3 = _tts_wrap("Replace '\u043f\u0440\u0438\u0432\u0435\u0442' with '\u0437\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439\u0442\u0435'.")
t4 = _tts_wrap("Замени слово привет на здравствуйте.")
print("wrap bare   :", t1[:80])
print("wrap templ. :", t2[:80])
print("wrap replace:", t3[:80])
print("wrap ru-edit:", t4[:80])
