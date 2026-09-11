from datetime import date

from dsg_compliance.chunking import chunk_articles, chunk_decisions
from dsg_compliance.ingestion.edoeb import Decision
from dsg_compliance.ingestion.fedlex import Article


def _article(law_id="dsg", number="6", heading="Grundsaetze", text="Wer Personendaten bearbeitet...") -> Article:
    return Article(
        law_id=law_id,
        law_title="Bundesgesetz ueber den Datenschutz (DSG)",
        article_number=number,
        heading=heading,
        text=text,
        source_url="https://www.fedlex.admin.ch/eli/cc/2022/491/de",
    )


def _decision(pdf_url="https://www.edoeb.admin.ch/decisions/example.pdf", full_text="Text.", law_version="DSG (ab 01.09.2023)") -> Decision:
    return Decision(
        title="Beispielverfuegung",
        description="Art. 6 DSG - Einhaltung der Datenbearbeitungsgrundsaetze",
        publication_date=date(2024, 3, 1),
        pdf_url=pdf_url,
        law_version=law_version,
        full_text=full_text,
    )


class TestChunkArticles:
    def test_basic_fields_map_through(self):
        [chunk] = chunk_articles([_article()])
        assert chunk.chunk_id == "dsg-art6"
        assert chunk.source_type == "statute"
        assert chunk.article_number == "6"
        assert chunk.law_version == "DSG (ab 01.09.2023)"
        assert "Art. 6 Grundsaetze" in chunk.text

    def test_repeated_article_number_gets_disambiguating_suffix(self):
        # e.g. a law's "Aenderung anderer Erlasse" annex reusing "Art. 3"
        # for a different, amended law - see chunk_articles docstring.
        articles = [_article(number="3"), _article(number="3"), _article(number="3")]
        chunks = chunk_articles(articles)
        ids = [c.chunk_id for c in chunks]
        assert ids == ["dsg-art3", "dsg-art3-dup1", "dsg-art3-dup2"]
        # every occurrence is kept, none silently dropped
        assert len(chunks) == 3

    def test_different_laws_do_not_collide(self):
        chunks = chunk_articles([_article(law_id="dsg", number="6"), _article(law_id="dsv", number="6")])
        assert {c.chunk_id for c in chunks} == {"dsg-art6", "dsv-art6"}

    def test_custom_law_version_is_applied(self):
        [chunk] = chunk_articles([_article()], law_version="aDSG (bis 31.08.2023)")
        assert chunk.law_version == "aDSG (bis 31.08.2023)"


class TestChunkDecisions:
    def test_skips_decisions_without_full_text(self):
        d = _decision(full_text=None)
        assert chunk_decisions([d]) == []

    def test_short_decision_produces_one_chunk_with_expected_metadata(self):
        d = _decision(full_text="Kurzer Entscheidtext.")
        [chunk] = chunk_decisions([d])
        assert chunk.source_type == "decision"
        assert chunk.source_title == "Beispielverfuegung"
        assert chunk.law_version == "DSG (ab 01.09.2023)"
        assert chunk.decision_date == date(2024, 3, 1)
        assert chunk.decision_summary == d.description
        assert chunk.text == "Kurzer Entscheidtext."

    def test_long_decision_is_split_into_multiple_overlapping_chunks(self):
        # Paragraphs long enough to force more than one chunk at max_chars=50.
        paragraphs = [f"Absatz Nummer {i} mit etwas mehr Fuelltext darin." for i in range(10)]
        d = _decision(full_text="\n".join(paragraphs))
        chunks = chunk_decisions([d], max_chars=50, overlap=10)
        assert len(chunks) > 1
        # consecutive chunks share the declared id prefix and are ordered
        for i, c in enumerate(chunks):
            assert c.chunk_id.startswith("decision-")
            assert c.chunk_id.endswith(f"-{i}")

    def test_chunk_id_is_stable_across_process_restarts(self):
        """Regression test: chunk_id must not depend on Python's built-in
        hash(), which is randomly salted per-process (PYTHONHASHSEED) and
        would silently break vector_store.upsert_chunks's id-based dedup on
        every re-ingestion run (see chunk_decisions's comment)."""
        import os
        import subprocess
        import sys

        script = (
            "from dsg_compliance.chunking import chunk_decisions;"
            "from dsg_compliance.ingestion.edoeb import Decision;"
            "from datetime import date;"
            "d = Decision(title='t', description='', publication_date=date(2024,1,1),"
            " pdf_url='https://www.edoeb.admin.ch/decisions/example.pdf',"
            " law_version='DSG (ab 01.09.2023)', full_text='Text.');"
            "print(chunk_decisions([d])[0].chunk_id)"
        )
        env1 = {**os.environ, "PYTHONHASHSEED": "1"}
        env2 = {**os.environ, "PYTHONHASHSEED": "2"}
        first = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env1,
        ).stdout.strip()
        second = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env2,
        ).stdout.strip()
        assert first == second, (
            "chunk_id changed between two different PYTHONHASHSEED values - "
            "it depends on an unsalted-assumed hash that isn't actually stable"
        )

    def test_two_decisions_get_distinct_id_prefixes(self):
        chunks = chunk_decisions([
            _decision(pdf_url="https://www.edoeb.admin.ch/a.pdf"),
            _decision(pdf_url="https://www.edoeb.admin.ch/b.pdf"),
        ])
        assert chunks[0].chunk_id != chunks[1].chunk_id
