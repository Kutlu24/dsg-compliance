"""FastAPI/CLI entry points. Run with:
    python -m dsg_compliance.cli ingest
    python -m dsg_compliance.cli ask "..."
    python -m dsg_compliance.cli build-citation-graph
"""
from __future__ import annotations

import json

import click
import yaml

from .chunking import chunk_articles, chunk_decisions
from .config import config_dir, data_dir
from .embeddings import embed_texts
from .ingestion.edoeb import EDOEB_ADSG_URL, EDOEB_VERFUEGUNGEN_URL, ingest_decisions
from .ingestion.fedlex import ingest_statute
from .rag import vector_store
from .rag.chat import ask as ask_question


@click.group()
def cli():
    pass


@cli.command("ingest")
@click.option("--statutes/--no-statutes", default=True)
@click.option("--decisions/--no-decisions", default=True)
@click.option(
    "--max-chunks",
    default=None,
    type=int,
    help="Cap total chunks sent to embed_texts, keeping all statute chunks "
    "and truncating decision chunks first. Gemini's free-tier embed_content "
    "quota is 1000 requests/day (per embedded text, not per API call), well "
    "under this corpus's ~3100 chunks - use this to fit within budget.",
)
def ingest(statutes: bool, decisions: bool, max_chunks: int | None):
    """Fetches DSG/DSV + EDOEB decisions, chunks, embeds, and stores them
    in the local Chroma collection. Safe to re-run - upsert overwrites
    matching chunk_ids rather than duplicating."""
    statute_chunks: list = []
    decision_chunks: list = []

    if statutes:
        sources = yaml.safe_load((config_dir() / "sources.yaml").read_text(encoding="utf-8"))
        for law in sources["statute"]:
            # url_de looks like https://www.fedlex.admin.ch/eli/cc/{year}/{num}/de
            parts = law["url_de"].rstrip("/").split("/")
            lang, num, year = parts[-1], parts[-2], parts[-3]
            click.echo(f"Fetching statute {law['id']} (eli/cc/{year}/{num})...")
            articles = ingest_statute(
                law_id=law["id"],
                law_title=law["title"],
                year=int(year),
                num=int(num),
                lang=lang,
                display_url=law["url_de"],
            )
            click.echo(f"  {len(articles)} articles")
            statute_chunks += chunk_articles(articles)

    if decisions:
        click.echo("Fetching current-DSG EDOEB decisions...")
        current = ingest_decisions(EDOEB_VERFUEGUNGEN_URL, with_text=True)
        click.echo(f"  {len(current)} decisions")
        click.echo("Fetching aDSG (pre-revision) EDOEB decisions...")
        adsg = ingest_decisions(EDOEB_ADSG_URL, with_text=True)
        click.echo(f"  {len(adsg)} decisions")
        decision_chunks = chunk_decisions(current + adsg)

    all_chunks = statute_chunks + decision_chunks
    already = vector_store.existing_ids()
    new_chunks = [c for c in all_chunks if c.chunk_id not in already]
    click.echo(
        f"\n{len(all_chunks)} chunks total, {len(already)} already embedded, "
        f"{len(new_chunks)} new."
    )

    if max_chunks is not None and len(new_chunks) > max_chunks:
        new_statute = [c for c in new_chunks if c.source_type == "statute"]
        new_decision = [c for c in new_chunks if c.source_type != "statute"]
        keep_decisions = max(0, max_chunks - len(new_statute))
        new_chunks = new_statute + new_decision[:keep_decisions]
        click.echo(
            f"Truncated to {len(new_chunks)} new chunks ({len(new_statute)} statute + "
            f"{len(new_chunks) - len(new_statute)} decision) to fit --max-chunks={max_chunks}. "
            "Re-run the same command later (e.g. once the daily embedding quota resets) "
            "to embed the rest - already-embedded chunks are skipped automatically."
        )

    if not new_chunks:
        click.echo("Nothing new to embed.")
        return

    click.echo(f"Embedding {len(new_chunks)} chunks via Gemini API...")
    vectors = embed_texts([c.text for c in new_chunks], task_type="RETRIEVAL_DOCUMENT")
    vector_store.upsert_chunks(new_chunks, vectors)
    click.echo(f"Stored. Collection now has {vector_store.count()} chunks.")


@cli.command("query")
@click.argument("question")
@click.option("--top-k", default=5, type=int)
def query_cmd(question: str, top_k: int):
    """Retrieval-only smoke test - no LLM yet, just shows what would be
    retrieved for a question, to sanity-check the index before building
    the chat layer on top of it."""
    [vector] = embed_texts([question], task_type="RETRIEVAL_QUERY")
    hits = vector_store.query(vector, top_k=top_k)
    for h in hits:
        m = h["metadata"]
        label = (
            f"Art. {m.get('article_number')} {m['source_title']}"
            if m["source_type"] == "statute"
            else f"{m['source_title']} ({m.get('decision_date', '?')})"
        )
        click.echo(f"\n[{m['law_version']}] {label}  (distance={h['distance']:.3f})")
        click.echo(f"  {h['text'][:300].replace(chr(10), ' ')}")
        click.echo(f"  -> {m['source_url']}")


@cli.command("ask")
@click.argument("question")
@click.option("--top-k", default=5, type=int)
def ask_cmd(question: str, top_k: int):
    """Retrieval + cited answer generation (see rag/chat.py for the
    provider used - GLM by default, see .env)."""
    result = ask_question(question, top_k=top_k)
    click.echo(f"\n{result.answer}\n")
    click.echo(f"(Modell: {result.model_used})")
    click.echo("Quellen:")
    for s in result.sources:
        click.echo(f"  - [{s.law_version}] {s.label} (distance={s.distance:.3f}) -> {s.source_url}")


@cli.command("build-citation-graph")
@click.option(
    "--out",
    default=None,
    help="Output JSON path (default: data/analysis/citation_graph.json).",
)
def build_citation_graph_cmd(out: str | None):
    """Fetches every EDOEB decision (current + aDSG) and extracts the BGE
    (Federal Supreme Court) case citations from each one, producing a
    bipartite decision<->precedent citation graph. See analysis/citations.py
    for why this replaced the originally-proposed EDOEB-to-EDOEB citation
    network (that one doesn't exist in this corpus - decisions don't cite
    each other)."""
    from .analysis.citations import build_citation_graph

    click.echo("Fetching current-DSG EDOEB decisions...")
    current = ingest_decisions(EDOEB_VERFUEGUNGEN_URL, with_text=True)
    click.echo(f"  {len(current)} decisions")
    click.echo("Fetching aDSG (pre-revision) EDOEB decisions...")
    adsg = ingest_decisions(EDOEB_ADSG_URL, with_text=True)
    click.echo(f"  {len(adsg)} decisions")

    graph = build_citation_graph(current + adsg)
    n_with_citations = sum(1 for d in graph.decisions if d.bge_citations)
    click.echo(
        f"\n{n_with_citations}/{len(graph.decisions)} decisions cite at least one BGE ruling."
    )
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


if __name__ == "__main__":
    cli()
