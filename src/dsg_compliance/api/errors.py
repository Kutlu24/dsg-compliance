"""Maps exceptions actually seen from the chat providers (GLM/Gemini/
Anthropic) to a clear HTTP status + message, instead of FastAPI's default
opaque 500 - same rationale as job-suche/src/job_suche/api/errors.py.

`lang` picks the message language explicitly (mirrors AskRequest.lang in
api/app.py) rather than guessing from the request - DSG's UI defaults to
German, DSA's to English, and this error path is hit before any answer
text exists to infer a language from."""
from __future__ import annotations

from fastapi import HTTPException

_MESSAGES = {
    "de": {
        "rate_limit": "Der KI-Dienst hat sein Tages-/Ratenlimit erreicht. Bitte spaeter erneut "
        "versuchen (siehe config.py fuer den aktuellen CHAT_PROVIDER).",
        "config": "Konfigurationsfehler: {msg}",
        "unexpected": "Unerwarteter Fehler ({name}): {msg}",
    },
    "en": {
        "rate_limit": "The AI service has hit its rate/quota limit. Please try again later "
        "(see config.py for the current CHAT_PROVIDER).",
        "config": "Configuration error: {msg}",
        "unexpected": "Unexpected error ({name}): {msg}",
    },
}


def friendly_llm_error(exc: Exception, lang: str = "de") -> HTTPException:
    name = type(exc).__name__
    msg = str(exc)
    strings = _MESSAGES.get(lang, _MESSAGES["de"])

    if "RateLimit" in name or "429" in msg or "quota" in msg.lower():
        return HTTPException(503, strings["rate_limit"])
    if isinstance(exc, RuntimeError) and "API_KEY" in msg:
        return HTTPException(500, strings["config"].format(msg=msg))
    return HTTPException(500, strings["unexpected"].format(name=name, msg=msg[:200]))
