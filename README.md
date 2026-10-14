# Project 8 - Model Governance Review Assistant

Document intelligence system for reviewing complex governance PDFs and producing grounded, evidence-linked insights.

## What It Does

- Upload one or more governance PDFs.
- Convert every page into text.
- Detect pages likely to contain tables, figures, graphs, or images using text/layout signals.
- Extract visual evidence only from those likely pages.
- Chunk parsed content into retrieval-ready passages.
- Answer user questions with:
  - grounded text evidence,
  - page citations,
  - matched table/figure/image evidence when relevant.

## Project Shape

```text
src/docgov/
  app.py                 Streamlit UI
  pipeline.py            End-to-end orchestration
  pdf_parser.py          PDF page text and visual extraction
  page_classifier.py     Regex/layout heuristics for visual-heavy pages
  retrieval.py           Local TF-IDF retrieval
  generation.py          Grounded answer generation interface
  models.py              Shared data models
tests/
  test_page_classifier.py
  test_retrieval.py
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
streamlit run src/docgov/app.py
```

Run the backend first:

```bash
uvicorn docgov.api:app --reload
```

Then run the UI in a second terminal:

```bash
streamlit run src/docgov/app.py
```

## Current Extraction Strategy

1. Parse text from every page.
2. Mark likely visual/table pages using regex cues such as `table`, `figure`, `chart`, `graph`, plus layout cues like dense numeric rows.
3. For marked pages, crop detected images from the PDF and store them as evidence assets.
4. Create image/table evidence records using nearby page text as searchable context.
5. Retrieve relevant text chunks for a question.
6. Retrieve relevant visual evidence against the same question and retrieved text.
7. Generate a grounded response with citations.

The default answer generator is deterministic and local. Replace `LocalGroundedGenerator` in `generation.py` with an LLM-backed implementation when you are ready.
