"""Check lexicon respellings through the product path."""
import sys

sys.path.insert(0, r"G:\AI\AuK\src")
from auk.infer.infer_gradio import _accentize_ru

for s in [
    "Электронное объявление появилось на сайте вчера.",
    "Объявления и объявлений было много.",
    "Администратор объяснил правила пользования сервисом.",
    "Цель проекта — устойчивое развитие региона.",
    "Пожалуйста, не опаздывайте на встречу.",
    "Встреча с друзьями перенесена на четверг.",
    "Сохранение энергии важно.",
    "Законы сохранения материи.",
    "Мне нравится смотреть на звёзды.",
    "Энергия ветра и энергия солнца.",
]:
    print(repr(_accentize_ru(s)).encode("ascii", "backslashreplace").decode())
