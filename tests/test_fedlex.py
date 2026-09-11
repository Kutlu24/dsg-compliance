from dsg_compliance.ingestion.fedlex import parse_articles

_SAMPLE_TEXT = """\
Bundesgesetz ueber den Datenschutz (DSG)

1. Kapitel: Allgemeine Bestimmungen

Art. 1 Zweck
Dieses Gesetz bezweckt den Schutz der Persoenlichkeit und der Grundrechte von
natuerlichen Personen, ueber die Personendaten bearbeitet werden.

Art. 5 Begriffe
In diesem Gesetz bedeuten:
a. Personendaten: alle Angaben, die sich auf eine bestimmte oder bestimmbare
   natuerliche Person beziehen;

Art. 6a Bearbeitungsgrundsaetze
1 Personendaten muessen rechtmaessig bearbeitet werden.
"""


def test_splits_into_one_article_per_heading():
    articles = parse_articles(_SAMPLE_TEXT, law_id="dsg", law_title="DSG", source_url="https://example.org/dsg")
    assert [a.article_number for a in articles] == ["1", "5", "6a"]


def test_heading_and_body_are_captured_separately():
    [art1, *_] = parse_articles(_SAMPLE_TEXT, law_id="dsg", law_title="DSG", source_url="https://example.org/dsg")
    assert art1.heading == "Zweck"
    assert "Schutz der Persoenlichkeit" in art1.text
    # the "Art. 1 Zweck" header line itself is consumed by the regex match,
    # so the body text should not start with a repeat of the heading
    assert not art1.text.startswith("Zweck")


def test_article_number_with_letter_suffix_is_supported():
    articles = parse_articles(_SAMPLE_TEXT, law_id="dsg", law_title="DSG", source_url="https://example.org/dsg")
    art6a = next(a for a in articles if a.article_number == "6a")
    assert "Bearbeitungsgrundsaetze" == art6a.heading


def test_last_article_body_runs_to_end_of_text():
    articles = parse_articles(_SAMPLE_TEXT, law_id="dsg", law_title="DSG", source_url="https://example.org/dsg")
    assert "rechtmaessig bearbeitet werden" in articles[-1].text


def test_no_article_headers_returns_empty_list():
    assert parse_articles("Kein Artikel hier.", law_id="dsg", law_title="DSG", source_url="https://example.org/dsg") == []


def test_law_id_title_and_source_url_propagate_to_every_article():
    articles = parse_articles(_SAMPLE_TEXT, law_id="dsg", law_title="Bundesgesetz ueber den Datenschutz", source_url="https://example.org/dsg")
    assert all(a.law_id == "dsg" for a in articles)
    assert all(a.law_title == "Bundesgesetz ueber den Datenschutz" for a in articles)
    assert all(a.source_url == "https://example.org/dsg" for a in articles)
