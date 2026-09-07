import click
import yaml

from .chunking import chunk_articles, chunk_decisions
from .config import config_dir
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
def ingest(statutes: bool, decisions: bool):
    """Fetches DSG/DSV + EDOEB decisions, chunks, embeds, and stores them
    in the local Chroma collection. Safe to re-run - upsert overwrites
    matching chunk_ids rather than duplicating."""
    all_chunks = []

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
            all_chunks += chunk_articles(articles)

    if decisions:
        click.echo("Fetching current-DSG EDOEB decisions...")
        current = ingest_decisions(EDOEB_VERFUEGUNGEN_URL, with_text=True)
        click.echo(f"  {len(current)} decisions")
        click.echo("Fetching aDSG (pre-revision) EDOEB decisions...")
        adsg = ingest_decisions(EDOEB_ADSG_URL, with_text=True)
        click.echo(f"  {len(adsg)} decisions")
        all_chunks += chunk_decisions(current + adsg)

    click.echo(f"\n{len(all_chunks)} chunks total. Embedding (local model, first run downloads the model)...")
    vectors = embed_texts([c.text for c in all_chunks])
    vector_store.upsert_chunks(all_chunks, vectors)
    click.echo(f"Stored. Collection now has {vector_store.count()} chunks.")


@cli.command("query")
@click.argument("question")
@click.option("--top-k", default=5, type=int)
def query_cmd(question: str, top_k: int):
    """Retrieval-only smoke test - no LLM yet, just shows what would be
    retrieved for a question, to sanity-check the index before building
    the chat layer on top of it."""
    [vector] = embed_texts([question])
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


if __name__ == "__main__":
    cli()
