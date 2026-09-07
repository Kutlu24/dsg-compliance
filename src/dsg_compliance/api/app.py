"""FastAPI backend for the DSG Compliance web UI. Run with:
    uvicorn dsg_compliance.api.app:app --reload

Endpoints:
    POST /ask     {question, top_k?} -> AnswerWithSources (see rag/chat.py)
    GET  /config  -> {provider, model} - which LLM is actually answering
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import settings
from ..rag.chat import AnswerWithSources, ask as ask_question
from .errors import friendly_llm_error

app = FastAPI(title="DSG Compliance")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="ui")


class ConfigInfo(BaseModel):
    provider: str
    model: str


_MODEL_BY_PROVIDER = {
    "glm": lambda: settings.glm_model,
    "gemini": lambda: settings.gemini_model,
    "anthropic": lambda: "claude-opus-5",
}


@app.get("/config", response_model=ConfigInfo)
def get_config() -> ConfigInfo:
    return ConfigInfo(
        provider=settings.chat_provider,
        model=_MODEL_BY_PROVIDER[settings.chat_provider](),
    )


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/ask", response_model=AnswerWithSources)
def ask_endpoint(req: AskRequest) -> AnswerWithSources:
    try:
        return ask_question(req.question, top_k=req.top_k)
    except Exception as e:
        raise friendly_llm_error(e) from e
