# Swiss Compliance Assistant (DSG + DSA)

A RAG-grounded chatbot for two Swiss-relevant compliance questions:

- **DSG** (Datenschutzgesetz, Switzerland's revised Federal Act on Data Protection, in force since 1 September 2023) — for Swiss-based researchers/businesses checking their own data-handling practices.
- **DSA** (EU Digital Services Act) — for Swiss businesses assessing whether the DSA's extraterritorial scope reaches them.

Every answer cites the specific statute article, ordinance clause, or official guidance passage it's based on. The two domains used to be separate deployments (`dsg-compliance`, `dsa-compliance`); merged into this one app/page because they're the exact same RAG architecture over a different corpus (see [Architecture](#architecture)).

🇩🇪 German version: [README.de.md](README.de.md)

## Why this exists

There are Swiss "private AI" hosting services (run your chatbot on Swiss soil, DSG-compliant *infrastructure*) and general compliance-consulting content, but nothing found that lets someone actually *ask the law itself* — ingesting the statute text and official guidance/decisions into a RAG index, so questions get answered with a citation back to source, not a hosting company's marketing copy.

**This is an informational aid, not legal advice.** It answers by retrieving and citing the actual source text; it does not replace consulting a lawyer, the FDPIC, or the European Commission for a real compliance decision.

## Scope

- **DSG**: the statute (SR 235.1) and its implementing ordinance (DSV, SR 235.11), plus FDPIC (EDÖB) guidance and published decisions (current + pre-revision aDSG, kept distinct — see the "current law vs. old law" badges in the UI). Sources: [`config/sources.yaml`](config/sources.yaml).
- **DSA**: the regulation text (EUR-Lex), European Commission guidance, and Swiss-relevance commentary. No EDÖB-equivalent published-decision archive exists for DSA yet. Sources: [`config/sources_dsa.yaml`](config/sources_dsa.yaml).

## Architecture

Both domains share one pipeline shape, one Chroma store, and one deployment - but are **not** forced into one shared schema, because the legal domains genuinely differ (DSG's current-law/old-law distinction and decision metadata has no DSA equivalent):

```
src/dsg_compliance/
  config.py, embeddings.py        <- shared: settings, Gemini embedding calls
  domains/
    dsg/  chunking.py, ingestion/{fedlex,edoeb}.py, rag/{vector_store,chat,faithfulness}.py, analysis/citations.py
    dsa/  chunking.py, ingestion/{eurlex,guidance}.py, rag/{vector_store,chat,faithfulness}.py
  api/app.py   <- one FastAPI app; POST /ask takes {question, domain: "dsg"|"dsa", lang}
  cli.py       <- one CLI; every command takes --domain dsg|dsa
```

Each domain's Chroma collection (`dsg_compliance`, `dsa_compliance`) lives in the same `data/chroma/` persist directory - two named collections in one store, no conflict. The frontend (`frontend/index.html`) is one page with a DSG/DSA/"+"(both) toggle:

```
fedlex.admin.ch / eurlex.europa.eu ─┐
edoeb.admin.ch / EC guidance        ─┼─> ingestion ─> chunking ─> embeddings (Gemini API) ─> Chroma (2 collections)
                                     ┘                                                  │
                                                              query ─> retrieval per domain ┘─> LLM (GLM) ─> cited answer
```

**"Both" mode is not a third backend domain** - it fires one `/ask` per real domain in parallel (`Promise.all`) and shows both answers stacked, each still using its own domain's citation format. This deliberately avoids blending the two domains' retrieval into a single answer: cross-domain retrieval risks citing the wrong regulation for a given claim.

## Tech stack

Python, [Chroma](https://www.trychroma.com/) for the vector store, Gemini's free-tier embedding API (`gemini-embedding-001`) for retrieval, GLM (z.ai, free tier) for cited-answer generation — swappable to Gemini/Anthropic via `CHAT_PROVIDER`, `httpx` + `beautifulsoup4`/`lxml` for scraping Fedlex/EUR-Lex HTML, `pypdf` for the PDFs (DSG statute text and FDPIC guidance).

Embeddings moved from a local `sentence-transformers` model to the hosted Gemini API after the local model (768-dim, but ~1.1GB with torch) reliably OOM-killed the process on Render's free 512MB-RAM plan on every `/ask` request. `GEMINI_API_KEY` is therefore always required, independent of `CHAT_PROVIDER`.

## Running it

```bash
pip install -e .
python -m dsg_compliance.cli ingest --domain dsg   # fetch + chunk + embed the DSG corpus via the Gemini API
python -m dsg_compliance.cli ingest --domain dsa   # same, for the DSA corpus
python -m dsg_compliance.cli ask --domain dsg "Wie lange darf ich fuer die Beantwortung eines Auskunftsgesuchs brauchen?"
uvicorn dsg_compliance.api.app:app --reload        # serves the UI at /ui/, both domains
```

`ask` retrieves the most relevant statute articles/decisions/guidance for one domain and has an LLM write a cited answer (never presented without its source list - see `domains/*/rag/chat.py`). `query` does retrieval only, no LLM, useful for sanity-checking an index.

On Render, `ingest` runs as part of `buildCommand` (see `render.yaml`) rather than being committed to the repo, so the vector store is rebuilt fresh on every deploy and a free-tier sleep/wake cycle never re-triggers it.

## Checking answers are actually faithful to their citations

`ask` always attaches a source list, but nothing previously checked whether the generated answer's claims actually match what those sources say — only a human reading both side by side. `python -m dsg_compliance.cli eval-faithfulness --domain dsg` (or `--domain dsa`, optionally with your own questions as arguments) runs a small set of representative questions through that domain's real pipeline and scores each answer with [ragas](https://github.com/explodinggradients/ragas)'s `Faithfulness` metric: it breaks the answer into individual statements and checks each one against the retrieved excerpts. The judge is the same GLM model already used for answer generation — no second API key.

Not the same thing as `domains/dsg/analysis/citations.py`'s BGE citation graph (that's corpus-level, "who cites whom", DSG-only); this is per-answer, "does this specific answer hold up against what it cites", for either domain.

Requires the `eval` extra: `pip install -e ".[eval]"` (pins `langchain-community==0.3.31` — see the extra's comment in `pyproject.toml` for why a newer version breaks `import ragas`). `tests/domains/{dsg,dsa}/test_faithfulness.py` regression-test the metric itself against a known-faithful and a known-fabricated example per domain; skipped automatically without `GLM_API_KEY`.

## Status

Working end to end for both domains: DSG (79 DSG + 47 DSV articles, 4 current + 71 aDSG EDOEB decisions, 900 chunks) and DSA (143 chunks, statute + EC/Swiss-relevance guidance), one shared Gemini-API embedding + Chroma store, and a cited-answer chat layer per domain (GLM by default - free tier, see `config.py` for why; swap to Gemini or Anthropic via `CHAT_PROVIDER` once available). Verified live via a real browser: DSG-only, DSA-only, and "both" mode all return correctly-cited, correctly-badged answers (2026-09-25).
