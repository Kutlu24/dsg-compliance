"""FastAPI backend for the Swiss Compliance Assistant - DSG (Swiss data
protection) and DSA (EU Digital Services Act) share this one app/deployment
rather than two, since they share the exact same RAG architecture (see
domains/dsg/ and domains/dsa/, and cli.py's module docstring for why the
two domains still keep separate chunking/rag modules rather than a forced
common schema). Run with:
    uvicorn dsg_compliance.api.app:app --reload

Endpoints:
    POST /ask             {question, domain, top_k?, lang?} -> AnswerWithSources
    GET  /config          -> {provider, model} - which LLM is actually answering
    GET  /citation-graph  -> dsg only, see domains/dsg/analysis/citations.py
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import data_dir, settings
from .errors import friendly_llm_error


def _allowed_origins() -> list[str]:
    """CORS allowlist: ALLOWED_ORIGINS env (comma-separated) plus this
    service's RENDER_EXTERNAL_URL; localhost only when not on Render."""
    origins = [
        o.strip()
        for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
        if o.strip()
    ]
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if render_url and render_url not in origins:
        origins.append(render_url)
    if not render_url and not origins:
        origins = [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:5173",
        ]
    return origins


app = FastAPI(title="Swiss Compliance Assistant (DSG + DSA)")
app.add_middleware(
    CORSMiddleware, allow_origins=_allowed_origins(), allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
    )
    return response


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
    # One chat_provider/model setting for both domains - they were already
    # sharing the same GLM/Gemini keys and free-tier quota before the
    # merge, so there is nothing domain-specific to report here.
    return ConfigInfo(
        provider=settings.chat_provider,
        model=_MODEL_BY_PROVIDER[settings.chat_provider](),
    )


def _load_citation_graph() -> dict:
    path = data_dir() / "analysis" / "citation_graph.json"
    if not path.exists():
        raise HTTPException(404, "citation_graph.json not built yet - run `dsg_compliance.cli build-citation-graph`")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/citation-graph")
def get_citation_graph():
    """dsg only - serves the pre-built EDOEB-decision -> BGE-precedent
    citation graph (see domains/dsg/analysis/citations.py and cli.py's
    build-citation-graph command for how it's generated). DSA has no
    equivalent published-decision archive, so there is no DSA version of
    this endpoint."""
    return _load_citation_graph()


@app.get("/export/decisions.json", include_in_schema=False)
def export_decisions_json() -> StreamingResponse:
    body = json.dumps(_load_citation_graph(), ensure_ascii=False, indent=2)
    return StreamingResponse(
        iter([body]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=dsg-compliance-decisions.json"},
    )


@app.get("/export/decisions.csv", include_in_schema=False)
def export_decisions_csv() -> StreamingResponse:
    graph = _load_citation_graph()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["title", "publication_date", "law_version", "pdf_url", "bge_citation_count", "bge_citations"])
    for d in graph["decisions"]:
        writer.writerow(
            [
                d["title"],
                d.get("publication_date") or "",
                d["law_version"],
                d["pdf_url"],
                len(d["bge_citations"]),
                "; ".join(d["bge_citations"]),
            ]
        )
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dsg-compliance-decisions.csv"},
    )


class AskRequest(BaseModel):
    question: str
    domain: Literal["dsg", "dsa"] = "dsg"
    top_k: int = 5
    lang: str = "de"  # "de" | "en" - explicit from the UI's language toggle,
    # not inferred from the question (the German-heavy DSG excerpts were
    # observed pulling weaker models into answering in German regardless).


@app.post("/ask", response_model=None)
def ask_endpoint(req: AskRequest):
    if req.domain == "dsg":
        from ..domains.dsg.rag.chat import ask as ask_question
    else:
        from ..domains.dsa.rag.chat import ask as ask_question
    try:
        return ask_question(req.question, top_k=req.top_k, lang=req.lang)
    except Exception as e:
        raise friendly_llm_error(e, lang=req.lang) from e
