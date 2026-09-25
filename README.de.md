# Swiss Compliance Assistant (DSG + DSA)

Ein RAG-gestützter Chatbot für zwei Compliance-Fragen mit Schweiz-Bezug:

- **DSG** (Bundesgesetz über den Datenschutz, revidiert, seit 1. September 2023 in Kraft) — für Forschende/Unternehmen in der Schweiz, die ihren eigenen Umgang mit Daten prüfen.
- **DSA** (EU Digital Services Act) — für Schweizer Unternehmen, die abklären, ob sie unter den extraterritorialen Geltungsbereich des DSA fallen.

Jede Antwort verweist auf den konkreten Gesetzesartikel, die Verordnungsbestimmung oder die offizielle Wegleitung, auf der sie beruht. Die beiden Bereiche waren früher getrennte Deployments (`dsg-compliance`, `dsa-compliance`); zusammengeführt in diese eine App/Seite, weil es genau dieselbe RAG-Architektur über einem anderen Korpus ist (siehe [Architektur](#architektur)).

🇬🇧 English version: [README.md](README.md)

## Warum dieses Projekt

Es gibt Schweizer „Private AI"-Hosting-Angebote (Chatbot auf Schweizer Boden, DSG-konforme *Infrastruktur*) und allgemeine Compliance-Beratungsinhalte — aber nichts Gefundenes, das erlaubt, das Gesetz selbst zu befragen: Gesetzestext und offizielle Wegleitungen/Entscheide werden in einen RAG-Index aufgenommen, sodass Antworten mit Quellenverweis kommen, nicht mit Marketingtext eines Hosting-Anbieters.

**Dies ist eine Informationshilfe, keine Rechtsberatung.** Es antwortet, indem es den tatsächlichen Quelltext abruft und zitiert; es ersetzt nicht die Konsultation eines Anwalts, des EDÖB oder der Europäischen Kommission für eine echte Compliance-Entscheidung.

## Umfang

- **DSG**: der Gesetzestext (SR 235.1) und die zugehörige Verordnung (DSV, SR 235.11), plus EDÖB-Wegleitungen und veröffentlichte Entscheide (aktuell + Alt-Recht aDSG, bewusst unterschieden — siehe die "aktuell/altes Recht"-Badges in der Oberfläche). Quellen: [`config/sources.yaml`](config/sources.yaml).
- **DSA**: der Verordnungstext (EUR-Lex), Wegleitungen der Europäischen Kommission und Schweiz-Bezug-Kommentare. Für DSA gibt es noch kein EDÖB-Äquivalent an veröffentlichten Entscheiden. Quellen: [`config/sources_dsa.yaml`](config/sources_dsa.yaml).

## Architektur

Beide Bereiche teilen sich eine Pipeline-Form, einen Chroma-Speicher und ein Deployment — aber **kein** erzwungenes gemeinsames Schema, weil sich die beiden Rechtsgebiete inhaltlich unterscheiden (DSG's Unterscheidung aktuelles/altes Recht und Entscheid-Metadaten haben im DSA keine Entsprechung):

```
src/dsg_compliance/
  config.py, embeddings.py        <- gemeinsam: Einstellungen, Gemini-Embedding-Aufrufe
  domains/
    dsg/  chunking.py, ingestion/{fedlex,edoeb}.py, rag/{vector_store,chat,faithfulness}.py, analysis/citations.py
    dsa/  chunking.py, ingestion/{eurlex,guidance}.py, rag/{vector_store,chat,faithfulness}.py
  api/app.py   <- eine FastAPI-App; POST /ask nimmt {question, domain: "dsg"|"dsa", lang}
  cli.py       <- eine CLI; jeder Befehl nimmt --domain dsg|dsa
```

Die Chroma-Collection jedes Bereichs (`dsg_compliance`, `dsa_compliance`) liegt im selben `data/chroma/`-Speicherverzeichnis — zwei benannte Collections in einem Speicher, kein Konflikt. Das Frontend (`frontend/index.html`) ist eine Seite mit einem DSG/DSA/"+"(beide)-Umschalter:

```
fedlex.admin.ch / eurlex.europa.eu ─┐
edoeb.admin.ch / EU-Wegleitungen    ─┼─> Ingestion ─> Chunking ─> Embeddings (Gemini-API) ─> Chroma (2 Collections)
                                     ┘                                                  │
                                                          Anfrage ─> Retrieval pro Bereich ┘─> LLM (GLM) ─> zitierte Antwort
```

**Der "Beide"-Modus ist kein dritter Backend-Bereich** — er feuert parallel je eine `/ask`-Anfrage pro echtem Bereich (`Promise.all`) und zeigt beide Antworten übereinander, jede weiterhin im Zitierformat ihres eigenen Bereichs. Das vermeidet bewusst, das Retrieval beider Bereiche in eine einzige Antwort zu vermischen: bereichsübergreifendes Retrieval riskiert, eine Aussage der falschen Verordnung zuzuschreiben.

## Technik

Python, [Chroma](https://www.trychroma.com/) als Vektordatenbank, Gemini's kostenlose Embedding-API (`gemini-embedding-001`) für das Retrieval, GLM (z.ai, kostenlose Stufe) für die zitierte Antwortgenerierung — umschaltbar auf Gemini/Anthropic über `CHAT_PROVIDER`, `httpx` + `beautifulsoup4`/`lxml` zum Scrapen der Fedlex-/EUR-Lex-HTML-Seiten, `pypdf` für die PDFs (DSG-Gesetzestext und EDÖB-Wegleitungen).

Die Embeddings wurden von einem lokalen `sentence-transformers`-Modell auf die gehostete Gemini-API umgestellt, nachdem das lokale Modell (768-dim, aber ~1,1GB mit torch) auf Renders kostenlosem 512MB-RAM-Plan bei jeder `/ask`-Anfrage den Prozess zuverlässig per OOM abgeschossen hat. `GEMINI_API_KEY` ist daher immer erforderlich, unabhängig von `CHAT_PROVIDER`.

## Ausführen

```bash
pip install -e .
python -m dsg_compliance.cli ingest --domain dsg   # DSG-Korpus abrufen + chunken + embedden via Gemini-API
python -m dsg_compliance.cli ingest --domain dsa   # dasselbe für den DSA-Korpus
python -m dsg_compliance.cli ask --domain dsg "Wie lange darf ich fuer die Beantwortung eines Auskunftsgesuchs brauchen?"
uvicorn dsg_compliance.api.app:app --reload        # bedient die Oberfläche unter /ui/, beide Bereiche
```

`ask` sucht die relevantesten Artikel/Entscheide/Wegleitungen eines Bereichs und lässt ein LLM eine zitierte Antwort schreiben (nie ohne die zugehörige Quellenliste — siehe `domains/*/rag/chat.py`). `query` macht nur das Retrieval, ohne LLM, nützlich um einen Index selbst zu prüfen.

Auf Render läuft `ingest` als Teil von `buildCommand` (siehe `render.yaml`), statt ins Repo committet zu werden — der Vektorspeicher wird so bei jedem Deploy frisch aufgebaut, und ein Schlaf-/Aufwach-Zyklus der kostenlosen Stufe löst ihn nie erneut aus.

## Prüfen, ob Antworten wirklich zu ihren Zitaten passen

`ask` liefert immer eine Quellenliste mit, aber bisher prüfte nichts automatisch, ob die generierte Antwort inhaltlich wirklich zu diesen Quellen passt — nur ein Mensch, der beides nebeneinander liest. `python -m dsg_compliance.cli eval-faithfulness --domain dsg` (oder `--domain dsa`, optional mit eigenen Fragen als Argumente) lässt ein paar repräsentative Fragen durch die echte Pipeline dieses Bereichs laufen und bewertet jede Antwort mit [ragas](https://github.com/explodinggradients/ragas)s `Faithfulness`-Metrik: sie zerlegt die Antwort in einzelne Aussagen und prüft jede gegen die abgerufenen Textstellen. Der Richter ist dasselbe GLM-Modell, das schon für die Antwortgenerierung verwendet wird — kein zweiter API-Key nötig.

Nicht dasselbe wie der BGE-Zitationsgraph in `domains/dsg/analysis/citations.py` (das ist korpusweit, "wer zitiert wen", nur DSG); dies hier ist pro Antwort, "hält diese konkrete Antwort dem stand, was sie zitiert", für beide Bereiche.

Benötigt das `eval`-Extra: `pip install -e ".[eval]"` (pinnt `langchain-community==0.3.31` — siehe der Kommentar zum Extra in `pyproject.toml`, warum eine neuere Version `import ragas` bricht). `tests/domains/{dsg,dsa}/test_faithfulness.py` testen die Metrik selbst pro Bereich gegen ein bekannt-treues und ein bekannt-erfundenes Beispiel; werden ohne `GLM_API_KEY` automatisch übersprungen.

## Status

Funktioniert Ende-zu-Ende für beide Bereiche: DSG (79 DSG- + 47 DSV-Artikel, 4 aktuelle + 71 aDSG-EDÖB-Entscheide, 900 Chunks) und DSA (143 Chunks, Verordnungstext + EU-/Schweiz-Wegleitungen), ein gemeinsamer Gemini-API-Embedding- + Chroma-Speicher, und eine zitierte Chat-Antwortebene pro Bereich (standardmässig GLM — kostenlose Stufe, siehe `config.py` für den Grund; umschaltbar auf Gemini oder Anthropic über `CHAT_PROVIDER`, sobald verfügbar). Live in einem echten Browser geprüft: Nur-DSG, Nur-DSA und "Beide"-Modus liefern alle korrekt zitierte, korrekt gebadgte Antworten (2026-09-25).
