from datetime import date

from dsg_compliance.analysis.citations import build_citation_graph, extract_bge_citations
from dsg_compliance.ingestion.edoeb import Decision


def _decision(full_text, pdf_url="https://www.edoeb.admin.ch/decisions/d.pdf") -> Decision:
    return Decision(
        title="Verfuegung",
        description="",
        publication_date=date(2024, 1, 1),
        pdf_url=pdf_url,
        law_version="DSG (ab 01.09.2023)",
        full_text=full_text,
    )


class TestExtractBgeCitations:
    def test_finds_a_single_citation(self):
        assert extract_bge_citations("Siehe BGE 136 II 508 fuer den massgebenden Grundsatz.") == ["BGE 136 II 508"]

    def test_finds_multiple_distinct_citations_in_order(self):
        text = "Vgl. BGE 138 II 346 sowie spaeter BGE 136 II 508."
        assert extract_bge_citations(text) == ["BGE 138 II 346", "BGE 136 II 508"]

    def test_deduplicates_repeated_citation(self):
        text = "BGE 136 II 508 ... spaeter erneut BGE 136 II 508 zitiert."
        assert extract_bge_citations(text) == ["BGE 136 II 508"]

    def test_tolerates_extra_whitespace_from_pdf_extraction_noise(self):
        assert extract_bge_citations("BGE  136   II   508") == ["BGE 136 II 508"]

    def test_no_citations_returns_empty_list(self):
        assert extract_bge_citations("Kein Bundesgerichtsentscheid hier erwaehnt.") == []


class TestBuildCitationGraph:
    def test_skips_decisions_without_full_text(self):
        graph = build_citation_graph([_decision(full_text=None)])
        assert graph.decisions == []
        assert graph.bge_frequency == {}

    def test_single_decision_with_one_citation(self):
        graph = build_citation_graph([_decision(full_text="Verweis auf BGE 136 II 508.")])
        assert len(graph.decisions) == 1
        assert graph.decisions[0].bge_citations == ["BGE 136 II 508"]
        assert graph.bge_frequency == {"BGE 136 II 508": 1}

    def test_frequency_counts_distinct_decisions_not_raw_mentions(self):
        # a citation mentioned twice within ONE decision should still only
        # count once in bge_frequency (frequency = number of decisions
        # citing it, not number of textual occurrences)
        d = _decision(full_text="BGE 136 II 508 ... und nochmals BGE 136 II 508.")
        graph = build_citation_graph([d])
        assert graph.bge_frequency == {"BGE 136 II 508": 1}

    def test_frequency_accumulates_across_multiple_decisions(self):
        decisions = [
            _decision(full_text="BGE 136 II 508.", pdf_url="https://example.org/a.pdf"),
            _decision(full_text="BGE 136 II 508 und BGE 138 II 346.", pdf_url="https://example.org/b.pdf"),
        ]
        graph = build_citation_graph(decisions)
        assert graph.bge_frequency == {"BGE 136 II 508": 2, "BGE 138 II 346": 1}

    def test_decision_metadata_is_preserved_in_output(self):
        d = _decision(full_text="Kein Zitat.", pdf_url="https://example.org/c.pdf")
        [entry] = build_citation_graph([d]).decisions
        assert entry.title == "Verfuegung"
        assert entry.pdf_url == "https://example.org/c.pdf"
        assert entry.law_version == "DSG (ab 01.09.2023)"
        assert entry.publication_date == "2024-01-01"
        assert entry.bge_citations == []
