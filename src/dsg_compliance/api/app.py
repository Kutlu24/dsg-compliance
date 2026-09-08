"""FastAPI backend for the DSG Compliance web UI. Run with:
    uvicorn dsg_compliance.api.app:app --reload

Endpoints:
    POST /ask     {question, top_k?} -> AnswerWithSources (see rag/chat.py)
    GET  /config  -> {provider, model} - which LLM is actually answering
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import data_dir, settings
from ..rag.chat import AnswerWithSources, ask as ask_question
from .errors import friendly_llm_error

app = FastAPI(title="DSG Compliance Assistant")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


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


@app.get("/citation-graph")
def get_citation_graph():
    """Serves the pre-built EDOEB-decision -> BGE-precedent citation graph
    (see analysis/citations.py and cli.py's build-citation-graph command
    for how it's generated - not computed on request)."""
    path = data_dir() / "analysis" / "citation_graph.json"
    if not path.exists():
        raise HTTPException(404, "citation_graph.json not built yet - run `dsg_compliance.cli build-citation-graph`")
    return json.loads(path.read_text(encoding="utf-8"))


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    lang: str = "de"  # "de" | "en" - which language the LLM must answer in,
    # set explicitly by the UI's language toggle rather than inferred from
    # the question text (unreliable: the German-heavy retrieved excerpts
    # were pulling weaker models into answering in German regardless).


@app.post("/ask", response_model=AnswerWithSources)
def ask_endpoint(req: AskRequest) -> AnswerWithSources:
    try:
        return ask_question(req.question, top_k=req.top_k, lang=req.lang)
    except Exception as e:
        raise friendly_llm_error(e) from e
