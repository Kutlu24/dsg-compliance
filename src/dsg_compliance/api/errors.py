"""Maps exceptions actually seen from the chat providers (GLM/Gemini/
Anthropic) to a clear HTTP status + message, instead of FastAPI's default
opaque 500 - same rationale as job-suche/src/job_suche/api/errors.py."""
from __future__ import annotations

from fastapi import HTTPException


def friendly_llm_error(exc: Exception) -> HTTPException:
    name = type(exc).__name__
    msg = str(exc)

    if "RateLimit" in name or "429" in msg or "quota" in msg.lower():
        return HTTPException(
            503,
            "Der KI-Dienst hat sein Tages-/Ratenlimit erreicht. Bitte spaeter erneut "
            "versuchen (siehe config.py fuer den aktuellen CHAT_PROVIDER).",
        )
    if isinstance(exc, RuntimeError) and "API_KEY" in msg:
        return HTTPException(500, f"Konfigurationsfehler: {msg}")
    return HTTPException(500, f"Unerwarteter Fehler ({name}): {msg[:200]}")
