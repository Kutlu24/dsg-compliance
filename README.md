# DSG Compliance

A RAG-grounded chatbot to help Swiss-based researchers check whether their own data-handling practices align with the **DSG** (Datenschutzgesetz, Switzerland's revised Federal Act on Data Protection, in force since 1 September 2023) — every answer cites the specific statute article, ordinance clause, or FDPIC guidance passage it's based on.

🇩🇪 German version: [README.de.md](README.de.md)

## Why this exists

There are Swiss "private AI" hosting services (run your chatbot on Swiss soil, DSG-compliant *infrastructure*) and general compliance-consulting content, but nothing found that lets a researcher actually *ask the law itself* — ingesting the statute, the ordinance, and the FDPIC's (EDÖB — Eidgenössischer Datenschutz- und Öffentlichkeitsbeauftragter) own guidance and case examples into a RAG index, so questions get answered with a citation back to source, not a hosting company's marketing copy.

**This is an informational aid, not legal advice.** It answers by retrieving and citing the actual source text; it does not replace consulting a lawyer or the FDPIC directly for a real compliance decision.

## Scope

- **Statute**: DSG (SR 235.1) and its implementing ordinance, the DSV (SR 235.11).
- **Guidance & case examples**: FDPIC guides (e.g. the TOM — technical/organizational measures — guide, the cross-border transfer guide) and its published recommendations/final reports, including historical ones from before the 2023 revision (still useful precedent).
- Real sources currently listed in [`config/sources.yaml`](config/sources.yaml) — every URL there was verified before being added. That file also lists known gaps (e.g. no current post-revision FDPIC case-example feed identified yet).

## Architecture

```
fedlex.admin.ch (DSG/DSV) ─┐
edoeb.admin.ch (guidance)  ─┼─> ingestion (fetch + parse) ─> chunking ─> embeddings (local) ─> Chroma
                            ┘                                                            │
                                                                       query ─> retrieval ┘─> LLM (GLM) ─> cited answer
```

## Tech stack

Python, [Chroma](https://www.trychroma.com/) for the vector store, local free `sentence-transformers` embeddings (no API key needed), GLM (z.ai, free tier) for cited-answer generation — swappable to Gemini/Anthropic via `CHAT_PROVIDER`, `httpx` + `beautifulsoup4`/`lxml` for scraping Fedlex HTML, `pypdf` for the PDFs (statute text and FDPIC guidance).

## Running it

```bash
pip install -e .
python -m dsg_compliance.cli ingest          # fetch + chunk + embed everything (~15 min, local CPU)
python -m dsg_compliance.cli ask "Wie lange darf ich fuer die Beantwortung eines Auskunftsgesuchs brauchen?"
```

`ask` retrieves the most relevant statute articles/decisions and has an LLM write a cited answer (never presented without its source list - see `rag/chat.py`). `query` does retrieval only, no LLM, useful for sanity-checking the index itself.

## Status

Working end to end: ingestion (79 DSG + 47 DSV articles, 4 current + 71 aDSG EDOEB decisions, 3138 chunks), local embedding + Chroma storage, and a cited-answer chat layer (GLM by default - free tier, see `config.py` for why; swap to Gemini or Anthropic via `CHAT_PROVIDER` once available).
