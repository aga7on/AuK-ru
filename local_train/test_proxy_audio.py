"""Test audio input through local Gemini proxy."""
import base64
import json
import sys
import urllib.request

KEY = os.environ.get("AUK_GEMINI_KEY", "")  # sanitized for public release
URL = "http://127.0.0.1:8045/v1/chat/completions"

model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.8-flash-medium"
wav = sys.argv[2] if len(sys.argv) > 2 else r"G:\AI\AuK\local_tests\voices_u18000\user_male_proiznoshenie.wav"

b64 = base64.b64encode(open(wav, "rb").read()).decode()
body = {
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": "РўСЂР°РЅСЃРєСЂРёР±РёСЂСѓР№ СЌС‚Рѕ Р°СѓРґРёРѕ РґРѕСЃР»РѕРІРЅРѕ РЅР° СЂСѓСЃСЃРєРѕРј. Р’РµСЂРЅРё С‚РѕР»СЊРєРѕ С‚РµРєСЃС‚."},
    ]}],
    "max_tokens": 300,
    "temperature": 0.0,
}
req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                             headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=120)
    data = json.loads(r.read().decode())
    msg = data["choices"][0]["message"]
    content = msg.get("content")
    if isinstance(content, list):
        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    print("MODEL:", model)
    print("REPLY:", str(content)[:400])
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:500])

