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

## Architektur (geplant)

```
fedlex.admin.ch (DSG/DSV) ─┐
edoeb.admin.ch (Wegleitungen) ─┼─> Ingestion (Abruf + Parsing) ─> Chunking ─> Embeddings ─> Chroma
                            ┘                                                            │
                                                                       Anfrage ─> Retrieval ┘─> Claude ─> zitierte Antwort
```

## Technik

Python, [Chroma](https://www.trychroma.com/) als Vektordatenbank, die Claude-API für Embeddings/Generierung, `httpx` + `beautifulsoup4`/`lxml` zum Scrapen der Fedlex-HTML-Seiten, `pypdf` für die PDF-Wegleitungen des EDÖB.

## Status

Nur Grundgerüst — Paketstruktur, Abhängigkeitsliste und eine verifizierte Quellenliste existieren; Ingestion, Chunking und die RAG-Abfrage-/Chat-Ebene sind noch nicht implementiert.
