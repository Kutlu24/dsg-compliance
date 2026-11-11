"""Limits in front of /ask and the CSV export. No model or index is used."""
import pytest
from fastapi.testclient import TestClient

from dsg_compliance.api import app as app_mod
from dsg_compliance.api import guard


@pytest.fixture
def c(monkeypatch):
    guard.limiter.reset()
    import dsg_compliance.domains.dsg.rag.chat as chat
    monkeypatch.setattr(chat, "ask", lambda q, top_k, lang: {"answer": "ok", "top_k": top_k, "lang": lang})
    return TestClient(app_mod.app)


Q = {"question": "Was ist ein Auskunftsbegehren?"}


def test_ask_is_rate_limited_per_client_and_globally(c, monkeypatch):
    monkeypatch.setitem(guard.RULES, "/ask", (2, 4))
    assert [c.post("/ask", json=Q).status_code for _ in range(3)] == [200, 200, 429]
    assert c.post("/ask", json=Q, headers={"X-Forwarded-For": "10.1.1.1, testclient"}).status_code == 429   # forged left-hand entry
    codes = [c.post("/ask", json=Q, headers={"X-Forwarded-For": f"198.51.100.{i}"}).status_code for i in range(4)]
    assert codes == [200, 200, 429, 429]                                                                      # global budget
    assert c.get("/config").status_code == 200


def test_ask_inputs_are_bounded_and_invalid_requests_are_not_charged(c, monkeypatch):
    monkeypatch.setitem(guard.RULES, "/ask", (2, 50))
    for bad in ({"question": "x" * 2001}, {"question": "ok ok", "top_k": 500}, {"question": "ok ok", "lang": "xx"}, {"question": ""}):
        assert c.post("/ask", json=bad).status_code == 422
    assert [c.post("/ask", json=Q).status_code for _ in range(3)] == [200, 200, 429]
    guard.limiter.reset()
    big = b'{"question": "' + b"x" * (guard.DEFAULT_BODY + 10) + b'"}'
    assert c.post("/ask", content=big, headers={"Content-Type": "application/json"}).status_code == 413
    assert c.post("/ask", content=(b"y" * 4096 for _ in range(30)), headers={"Content-Type": "application/json"}).status_code == 413


def test_csv_export_neutralises_formula_cells(c, monkeypatch):
    monkeypatch.setattr(app_mod, "_load_citation_graph", lambda: {"decisions": [
        {"title": '=HYPERLINK("http://evil.example","click")', "law_version": "nDSG", "pdf_url": "u", "bge_citations": []},
        {"title": "Normaler Titel", "law_version": "nDSG", "pdf_url": "u", "bge_citations": ["BGE 1"]}]})
    text = c.get("/export/decisions.csv").text
    assert "'=HYPERLINK" in text and "\n=HYPERLINK" not in text and "Normaler Titel" in text
