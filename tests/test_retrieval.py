from pathlib import Path

from docgov.models import TextChunk, VisualEvidence
from docgov.retrieval import RetrievalStore


def test_retrieval_returns_relevant_chunk() -> None:
    store = RetrievalStore(
        chunks=[
            TextChunk("c1", "doc", 1, "model validation policy approval committee"),
            TextChunk("c2", "doc", 2, "unrelated procurement workflow"),
        ],
        visuals=[],
    )

    results = store.search_chunks("validation committee")
    assert results[0].chunk.chunk_id == "c1"


def test_attach_visuals_to_selected_chunks_uses_fuzzy_source_text() -> None:
    store = RetrievalStore(
        chunks=[
            TextChunk("c1", "doc", 3, "model risk ratings are high for monitoring controls"),
        ],
        visuals=[
            VisualEvidence("v1", "doc", 3, "table", Path("table.png"), "Table of model risk ratings"),
            VisualEvidence("v2", "doc", 4, "figure", Path("figure.png"), "Architecture diagram"),
        ],
    )

    chunks = store.search_chunks("monitoring risk ratings")
    results = store.attach_visuals_to_chunks(chunks)
    assert results[0].evidence.evidence_id == "v1"
