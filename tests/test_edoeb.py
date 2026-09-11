from datetime import date

import httpx
import pytest

from dsg_compliance.ingestion import edoeb
from dsg_compliance.ingestion.edoeb import EDOEB_ADSG_URL, _parse_german_date, list_decisions

_SAMPLE_HTML = """
<html><body>
<h2 id="2024">2024</h2>
<ul class="list">
  <li><a class="download-item" href="https://www.edoeb.admin.ch/decisions/example1.pdf">
    <div>
      <h4 class="download-item__title">Verfuegung gegen Beispiel AG</h4>
      <p class="download-item__description">Art. 6 DSG - Einhaltung der Datenbearbeitungsgrundsaetze</p>
      <p class="download-item__meta-info">Publiziert am <span>3. März 2024</span></p>
    </div>
  </a></li>
  <li><a class="download-item" href="https://www.edoeb.admin.ch/decisions/example2.pdf">
    <div>
      <h4 class="download-item__title">Verfuegung ohne Datum</h4>
    </div>
  </a></li>
</ul>
</body></html>
"""


class TestParseGermanDate:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Publiziert am 3. Maerz 2024", None),  # "Maerz" (ASCII) isn't a key in _GERMAN_MONTHS ("März" is)
            ("Publiziert am 3. März 2024", date(2024, 3, 3)),
            ("am 15. Januar 2023 veroeffentlicht", date(2023, 1, 15)),
            ("31. Dezember 1999", date(1999, 12, 31)),
        ],
    )
    def test_known_formats(self, text, expected):
        assert _parse_german_date(text) == expected

    def test_no_date_in_text_returns_none(self):
        assert _parse_german_date("kein Datum hier") is None

    def test_unrecognized_month_name_returns_none(self):
        assert _parse_german_date("3. Foobar 2024") is None


class TestListDecisions:
    def test_parses_title_description_date_and_url(self, monkeypatch):
        def fake_get(url, timeout=None, follow_redirects=None, headers=None):
            return httpx.Response(200, text=_SAMPLE_HTML, request=httpx.Request("GET", url))

        monkeypatch.setattr(edoeb.httpx, "get", fake_get)
        decisions = list_decisions()

        assert len(decisions) == 2
        first = decisions[0]
        assert first.title == "Verfuegung gegen Beispiel AG"
        assert first.description == "Art. 6 DSG - Einhaltung der Datenbearbeitungsgrundsaetze"
        assert first.publication_date == date(2024, 3, 3)
        assert first.pdf_url == "https://www.edoeb.admin.ch/decisions/example1.pdf"
        assert first.law_version == "DSG (ab 01.09.2023)"

    def test_missing_optional_fields_default_sensibly(self, monkeypatch):
        def fake_get(url, timeout=None, follow_redirects=None, headers=None):
            return httpx.Response(200, text=_SAMPLE_HTML, request=httpx.Request("GET", url))

        monkeypatch.setattr(edoeb.httpx, "get", fake_get)
        second = list_decisions()[1]
        assert second.description == ""
        assert second.publication_date is None

    def test_adsg_url_is_tagged_with_old_law_version(self, monkeypatch):
        def fake_get(url, timeout=None, follow_redirects=None, headers=None):
            return httpx.Response(200, text=_SAMPLE_HTML, request=httpx.Request("GET", url))

        monkeypatch.setattr(edoeb.httpx, "get", fake_get)
        decisions = list_decisions(EDOEB_ADSG_URL)
        assert all(d.law_version == "aDSG (bis 31.08.2023)" for d in decisions)
