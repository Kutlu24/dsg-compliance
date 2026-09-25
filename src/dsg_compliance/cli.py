"""FastAPI/CLI entry points. Run with:
    python -m dsg_compliance.cli ingest --domain dsg
    python -m dsg_compliance.cli ingest --domain dsa
    python -m dsg_compliance.cli ask --domain dsg "..."
    python -m dsg_compliance.cli build-citation-graph   # dsg only
    python -m dsg_compliance.cli eval-faithfulness --domain dsg

Two independent legal-compliance domains (DSG - Swiss data protection,
DSA - EU Digital Services Act) share this one CLI/app/Chroma store rather
than being two separate deployments - see domains/dsg/ and domains/dsa/,
which mirror each other's module shape (chunking.py, ingestion/, rag/) but
are NOT unified into one shared implementation: the two legal domains have
genuinely different Chunk/SourceRef metadata (DSG's aDSG/current law-version
distinction has no DSA equivalent) and different system prompts, so forcing
a single schema would either lose DSG's law-version tracking or add unused
fields to DSA. What IS actually shared (config.py, embeddings.py, the
Chroma persist directory) lives at the package root.
"""
from __future__ import annotations

import click
import yaml

from .config import config_dir, data_dir


def _domain_module(domain: str):
    if domain == "dsg":
        from .domains import dsg

        return dsg
    if domain == "dsa":
        from .domains import dsa

        return dsa
    raise click.BadParameter(f"Unknown domain '{domain}' - must be 'dsg' or 'dsa'.")


@click.group()
def cli():
    pass


@cli.command("ingest")
@click.option("--domain", required=True, type=click.Choice(["dsg", "dsa"]))
@click.option("--statutes/--no-statutes", default=True)
@click.option("--secondary/--no-secondary", default=True, help="Decisions (dsg) or guidance (dsa).")
@click.option(
    "--max-chunks",
    default=None,
    type=int,
    help="Cap total chunks sent to embed_texts (dsg only - see domains/dsg's own "
    "ingest for why: Gemini's free-tier embed_content quota is 1000 requests/day).",
)
@click.option("--lang", default="ENG", help="dsa only: 3-letter CELEX language code for the statute text.")
def ingest(domain: str, statutes: bool, secondary: bool, max_chunks: int | None, lang: str):
    """Fetches a domain's statute + secondary sources, chunks, embeds, and
    stores them in that domain's Chroma collection (dsg_compliance /
    dsa_compliance - distinct collections in the one shared Chroma store,
    see domains/*/rag/vector_store.py). Safe to re-run - upsert overwrites
    matching chunk_ids rather than duplicating."""
    from .embeddings import embed_texts

    if domain == "dsg":
        from .domains.dsg import chunking
        from .domains.dsg.ingestion.edoeb import EDOEB_ADSG_URL, EDOEB_VERFUEGUNGEN_URL, ingest_decisions
        from .domains.dsg.ingestion.fedlex import ingest_statute
        from .domains.dsg.rag import vector_store

        statute_chunks: list = []
        decision_chunks: list = []
        if statutes:
            sources = yaml.safe_load((config_dir() / "sources.yaml").read_text(encoding="utf-8"))
            for law in sources["statute"]:
                parts = law["url_de"].rstrip("/").split("/")
                law_lang, num, year = parts[-1], parts[-2], parts[-3]
                click.echo(f"Fetching statute {law['id']} (eli/cc/{year}/{num})...")
                articles = ingest_statute(
                    law_id=law["id"], law_title=law["title"], year=int(year), num=int(num),
                    lang=law_lang, display_url=law["url_de"],
                )
                click.echo(f"  {len(articles)} articles")
                statute_chunks += chunking.chunk_articles(articles)
        if secondary:
            click.echo("Fetching current-DSG EDOEB decisions...")
            current = ingest_decisions(EDOEB_VERFUEGUNGEN_URL, with_text=True)
            click.echo(f"  {len(current)} decisions")
            click.echo("Fetching aDSG (pre-revision) EDOEB decisions...")
            adsg = ingest_decisions(EDOEB_ADSG_URL, with_text=True)
            click.echo(f"  {len(adsg)} decisions")
            decision_chunks = chunking.chunk_decisions(current + adsg)
        all_chunks = statute_chunks + decision_chunks
    else:
        from .domains.dsa import chunking
        from .domains.dsa.ingestion.eurlex import ingest_statute
        from .domains.dsa.ingestion.guidance import ingest_guidance_sources
        from .domains.dsa.rag import vector_store

        sources = yaml.safe_load((config_dir() / "sources_dsa.yaml").read_text(encoding="utf-8"))
        statute_chunks = []
        guidance_chunks = []
        if statutes:
            for law in sources["statute"]:
                click.echo(f"Fetching statute {law['id']} (CELEX {law['celex']}, {lang})...")
                articles = ingest_statute(
                    law_id=law["id"], law_title=law["title"], celex=law["celex"],
                    lang=lang, display_url=law.get("url_display"),
                )
                click.echo(f"  {len(articles)} articles")
                statute_chunks += chunking.chunk_articles(articles)
        if secondary:
            entries = sources.get("guidance", []) + sources.get("swiss_relevance", [])
            click.echo(f"Fetching {len(entries)} guidance/commentary pages...")
            pages = ingest_guidance_sources(entries)
            for p in pages:
                click.echo(f"  {p.source_id}: {len(p.text)} chars")
            guidance_chunks = chunking.chunk_guidance(pages)
        all_chunks = statute_chunks + guidance_chunks

    already = vector_store.existing_ids()
    new_chunks = [c for c in all_chunks if c.chunk_id not in already]
    click.echo(f"\n{len(all_chunks)} chunks total, {len(already)} already embedded, {len(new_chunks)} new.")

    if domain == "dsg" and max_chunks is not None and len(new_chunks) > max_chunks:
        new_statute = [c for c in new_chunks if c.source_type == "statute"]
        new_decision = [c for c in new_chunks if c.source_type != "statute"]
        keep_decisions = max(0, max_chunks - len(new_statute))
        new_chunks = new_statute + new_decision[:keep_decisions]
        click.echo(
            f"Truncated to {len(new_chunks)} new chunks ({len(new_statute)} statute + "
            f"{len(new_chunks) - len(new_statute)} decision) to fit --max-chunks={max_chunks}. "
            "Re-run later (e.g. once the daily embedding quota resets) - already-embedded "
            "chunks are skipped automatically."
        )

    if not new_chunks:
        click.echo("Nothing new to embed.")
        return

    click.echo(f"Embedding {len(new_chunks)} chunks via Gemini API...")
    vectors = embed_texts([c.text for c in new_chunks], task_type="RETRIEVAL_DOCUMENT")
    vector_store.upsert_chunks(new_chunks, vectors)
    click.echo(f"Stored. Collection now has {vector_store.count()} chunks.")


@cli.command("query")
@click.option("--domain", required=True, type=click.Choice(["dsg", "dsa"]))
@click.argument("question")
@click.option("--top-k", default=5, type=int)
def query_cmd(domain: str, question: str, top_k: int):
    """Retrieval-only smoke test for one domain - no LLM, just shows what
    would be retrieved for a question."""
    from .embeddings import embed_texts

    mod = _domain_module(domain)
    [vector] = embed_texts([question], task_type="RETRIEVAL_QUERY")
    hits = mod.rag.vector_store.query(vector, top_k=top_k)
    for h in hits:
        m = h["metadata"]
        if domain == "dsg":
            label = (
                f"Art. {m.get('article_number')} {m['source_title']}"
                if m["source_type"] == "statute"
                else f"{m['source_title']} ({m.get('decision_date', '?')})"
            )
            click.echo(f"\n[{m['law_version']}] {label}  (distance={h['distance']:.3f})")
        else:
            label = f"Article {m.get('article_number')} DSA" if m["source_type"] == "statute" else m["source_title"]
            click.echo(f"\n{label}  (distance={h['distance']:.3f})")
        click.echo(f"  {h['text'][:300].replace(chr(10), ' ')}")
        click.echo(f"  -> {m['source_url']}")


@cli.command("ask")
@click.option("--domain", required=True, type=click.Choice(["dsg", "dsa"]))
@click.argument("question")
@click.option("--top-k", default=5, type=int)
def ask_cmd(domain: str, question: str, top_k: int):
    """Retrieval + cited answer generation for one domain."""
    mod = _domain_module(domain)
    result = mod.rag.chat.ask(question, top_k=top_k)
    click.echo(f"\n{result.answer}\n")
    click.echo(f"(Model: {result.model_used})")
    click.echo("Sources:")
    for s in result.sources:
        prefix = f"[{s.law_version}] " if domain == "dsg" else ""
        click.echo(f"  - {prefix}{s.label} (distance={s.distance:.3f}) -> {s.source_url}")


@cli.command("build-citation-graph")
@click.option("--out", default=None, help="Output JSON path (default: data/analysis/citation_graph.json).")
def build_citation_graph_cmd(out: str | None):
    """dsg only: fetches every EDOEB decision (current + aDSG) and extracts
    the BGE (Federal Supreme Court) case citations from each one. See
    domains/dsg/analysis/citations.py for why this replaced the originally-
    proposed EDOEB-to-EDOEB citation network (decisions don't cite each
    other in this corpus)."""
    from .domains.dsg.analysis.citations import build_citation_graph
    from .domains.dsg.ingestion.edoeb import EDOEB_ADSG_URL, EDOEB_VERFUEGUNGEN_URL, ingest_decisions

    click.echo("Fetching current-DSG EDOEB decisions...")
    current = ingest_decisions(EDOEB_VERFUEGUNGEN_URL, with_text=True)
    click.echo(f"  {len(current)} decisions")
    click.echo("Fetching aDSG (pre-revision) EDOEB decisions...")
    adsg = ingest_decisions(EDOEB_ADSG_URL, with_text=True)
    click.echo(f"  {len(adsg)} decisions")

    graph = build_citation_graph(current + adsg)
    n_with_citations = sum(1 for d in graph.decisions if d.bge_citations)
    click.echo(f"\n{n_with_citations}/{len(graph.decisions)} decisions cite at least one BGE ruling.")
    click.echo(f"{len(graph.bge_frequency)} distinct BGE rulings cited.")
    top = sorted(graph.bge_frequency.items(), key=lambda kv: -kv[1])[:10]
    click.echo("Most-cited precedents:")
    for citation, n in top:
        click.echo(f"  {citation}: cited by {n} decision(s)")

    out_path = data_dir() / "analysis" / "citation_graph.json" if out is None else out
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"\nWritten to {out_path}")


@cli.command("eval-faithfulness")
@click.option("--domain", required=True, type=click.Choice(["dsg", "dsa"]))
@click.argument("questions", nargs=-1)
@click.option("--top-k", default=5, type=int)
@click.option("--threshold", default=0.8, type=float, help="Scores below this print as LOW.")
def eval_faithfulness_cmd(domain: str, questions: tuple[str, ...], top_k: int, threshold: float):
    """Runs each question through that domain's ask() and scores whether
    the generated answer's claims actually trace back to its cited
    sources, using ragas's Faithfulness metric. Requires the `eval` extra
    and GLM_API_KEY. Uses a small built-in set of representative questions
    if none are given on the command line."""
    import asyncio

    mod = _domain_module(domain)
    qs = list(questions) or mod.rag.faithfulness.DEFAULT_EVAL_QUESTIONS

    async def run() -> None:
        for q in qs:
            try:
                result = mod.rag.chat.ask(q, top_k=top_k)
                if not result.sources:
                    click.echo(f"[NO SOURCES] {q}")
                    continue
                score = await mod.rag.faithfulness.score_faithfulness(result)
            except Exception as e:
                click.echo(f"[ERROR] {q}\n        {type(e).__name__}: {e}")
                continue
            flag = "OK " if score >= threshold else "LOW"
            click.echo(f"[{flag} {score:.2f}] {q}")

    asyncio.run(run())


if __name__ == "__main__":
    cli()
