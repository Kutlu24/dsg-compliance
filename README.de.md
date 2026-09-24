# DSG Compliance

Ein RAG-gestützter Chatbot, der Forschenden in der Schweiz hilft zu prüfen, ob ihr eigener Umgang mit Daten dem **DSG** (Bundesgesetz über den Datenschutz, revidiert, seit 1. September 2023 in Kraft) entspricht — jede Antwort verweist auf den konkreten Gesetzesartikel, die Verordnungsbestimmung oder die EDÖB-Wegleitung, auf der sie beruht.

🇬🇧 English version: [README.md](README.md)

## Warum dieses Projekt

Es gibt Schweizer „Private AI"-Hosting-Angebote (Chatbot auf Schweizer Boden, DSG-konforme *Infrastruktur*) und allgemeine Compliance-Beratungsinhalte — aber nichts Gefundenes, das Forschenden erlaubt, das Gesetz selbst zu befragen: Gesetzestext, Verordnung und die eigenen Wegleitungen/Fallbeispiele des EDÖB (Eidgenössischer Datenschutz- und Öffentlichkeitsbeauftragter) werden in einen RAG-Index aufgenommen, sodass Antworten mit Quellenverweis kommen, nicht mit Marketingtext eines Hosting-Anbieters.

**Dies ist eine Informationshilfe, keine Rechtsberatung.** Es antwortet, indem es den tatsächlichen Quelltext abruft und zitiert; es ersetzt nicht die Konsultation eines Anwalts oder direkt des EDÖB für eine echte Compliance-Entscheidung.

## Umfang

- **Gesetzestext**: DSG (SR 235.1) und die zugehörige Verordnung, die DSV (SR 235.11).
- **Wegleitungen & Fallbeispiele**: EDÖB-Leitfäden (z. B. zu den technischen/organisatorischen Massnahmen (TOM), zur grenzüberschreitenden Datenübermittlung) sowie veröffentlichte Empfehlungen/Schlussberichte, einschliesslich historischer aus der Zeit vor der Revision 2023 (weiterhin als Präzedenz nützlich).
- Aktuelle, echte Quellen stehen in [`config/sources.yaml`](config/sources.yaml) — jede dort gelistete URL wurde vor der Aufnahme verifiziert. Diese Datei listet auch bekannte Lücken (z. B. wurde noch kein aktueller Fallbeispiel-Feed des EDÖB nach der Revision gefunden).

## Architektur

```
fedlex.admin.ch (DSG/DSV) ─┐
edoeb.admin.ch (Wegleitungen) ─┼─> Ingestion (Abruf + Parsing) ─> Chunking ─> Embeddings (Gemini-API) ─> Chroma
                            ┘                                                            │
                                                                       Anfrage ─> Retrieval ┘─> LLM (GLM) ─> zitierte Antwort
```

## Technik

Python, [Chroma](https://www.trychroma.com/) als Vektordatenbank, Gemini's kostenlose Embedding-API (`gemini-embedding-001`) für das Retrieval, GLM (z.ai, kostenlose Stufe) für die zitierte Antwortgenerierung — umschaltbar auf Gemini/Anthropic über `CHAT_PROVIDER`, `httpx` + `beautifulsoup4`/`lxml` zum Scrapen der Fedlex-HTML-Seiten, `pypdf` für die PDFs (Gesetzestext und EDÖB-Wegleitungen).

Die Embeddings wurden von einem lokalen `sentence-transformers`-Modell auf die gehostete Gemini-API umgestellt, nachdem das lokale Modell (768-dim, aber ~1,1GB mit torch) auf Renders kostenlosem 512MB-RAM-Plan bei jeder `/ask`-Anfrage den Prozess zuverlässig per OOM abgeschossen hat. `GEMINI_API_KEY` ist daher immer erforderlich, unabhängig von `CHAT_PROVIDER`.

## Ausführen

```bash
pip install -e .
python -m dsg_compliance.cli ingest          # alles abrufen + chunken + embedden via Gemini-API
python -m dsg_compliance.cli ask "Wie lange darf ich fuer die Beantwortung eines Auskunftsgesuchs brauchen?"
```

`ask` sucht die relevantesten Artikel/Entscheide und lässt ein LLM eine zitierte Antwort schreiben (nie ohne die zugehörige Quellenliste — siehe `rag/chat.py`). `query` macht nur das Retrieval, ohne LLM, nützlich um den Index selbst zu prüfen.

Auf Render läuft `ingest` als Teil von `buildCommand` (siehe `render.yaml`), statt ins Repo committet zu werden — der Vektorspeicher wird so bei jedem Deploy frisch aufgebaut, und ein Schlaf-/Aufwach-Zyklus der kostenlosen Stufe löst ihn nie erneut aus.

## Prüfen, ob Antworten wirklich zu ihren Zitaten passen

`ask` liefert immer eine Quellenliste mit, aber bisher prüfte nichts automatisch, ob die generierte Antwort inhaltlich wirklich zu diesen Quellen passt — nur ein Mensch, der beides nebeneinander liest. `python -m dsg_compliance.cli eval-faithfulness` (optional mit eigenen Fragen als Argumente) lässt ein paar repräsentative Fragen durch die echte Pipeline laufen und bewertet jede Antwort mit [ragas](https://github.com/explodinggradients/ragas)s `Faithfulness`-Metrik: sie zerlegt die Antwort in einzelne Aussagen und prüft jede gegen die abgerufenen Textstellen. Der Richter ist dasselbe GLM-Modell, das schon für die Antwortgenerierung verwendet wird — kein zweiter API-Key nötig.

Nicht dasselbe wie der BGE-Zitationsgraph in `analysis/citations.py` (das ist korpusweit, "wer zitiert wen"); dies hier ist pro Antwort, "hält diese konkrete Antwort dem stand, was sie zitiert."

Benötigt das `eval`-Extra: `pip install -e ".[eval]"` (pinnt `langchain-community==0.3.31` — siehe der Kommentar zum Extra in `pyproject.toml`, warum eine neuere Version `import ragas` bricht). `tests/test_faithfulness.py` testet die Metrik selbst gegen ein bekannt-treues und ein bekannt-erfundenes Beispiel; wird ohne `GLM_API_KEY` automatisch übersprungen.

## Status

Funktioniert Ende-zu-Ende: Ingestion (79 DSG- + 47 DSV-Artikel, 4 aktuelle + 71 aDSG-EDÖB-Entscheide, 3138 Chunks), Gemini-API-Embedding + Chroma-Speicherung, und eine zitierte Chat-Antwortebene (standardmässig GLM — kostenlose Stufe, siehe `config.py` für den Grund; umschaltbar auf Gemini oder Anthropic über `CHAT_PROVIDER`, sobald verfügbar).
