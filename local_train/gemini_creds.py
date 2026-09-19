"""Single credential provider for the local Gemini proxy. No secrets in new scripts.

Resolution order:
  1. env AUK_GEMINI_KEY
  2. env GEMINI_API_KEY / OPENAI_API_KEY
  3. the already-existing provider in local_train/auto_judge.py (imported, never printed)
Raises RuntimeError if nothing is available; the key is never logged.
"""
import importlib
import os


def get_key() -> str:
    for name in ("AUK_GEMINI_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v
    try:
        mod = importlib.import_module("auto_judge")
        v = getattr(mod, "KEY", "")
        if v:
            return v
    except Exception:
        pass
    raise RuntimeError("no Gemini proxy key: set AUK_GEMINI_KEY (see local_train/auto_judge.py)")


def get_url() -> str:
    return os.environ.get("AUK_GEMINI_URL", "http://127.0.0.1:8045/v1/chat/completions")


def get_model() -> str:
    return os.environ.get("AUK_GEMINI_MODEL", "gemini-3.8-flash-medium")
