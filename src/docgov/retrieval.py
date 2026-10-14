from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from uuid import uuid4

import chromadb

from .models import RetrievedChunk, RetrievedVisual, TextChunk, VisualEvidence


@dataclass
class RetrievalStore:
    chunks: list[TextChunk]
    visuals: list[VisualEvidence]

    def __post_init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(name=f"governance_chunks_{uuid4().hex}")
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        if self.chunks:
            self._collection.add(
                ids=[chunk.chunk_id for chunk in self.chunks],
                documents=[chunk.text for chunk in self.chunks],
                metadatas=[
                    {
                        "doc_id": chunk.doc_id,
                        "page_number": chunk.page_number,
                    }
                    for chunk in self.chunks
                ],
            )

    def search_chunks(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        results = self._collection.query(query_texts=[query], n_results=min(limit, len(self.chunks)))
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        retrieved: list[RetrievedChunk] = []
        for chunk_id, distance in zip(ids, distances):
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            score = 1.0 / (1.0 + float(distance))
            retrieved.append(RetrievedChunk(chunk, score))
        return retrieved

    def attach_visuals_to_chunks(
        self,
        retrieved_chunks: list[RetrievedChunk],
        limit: int = 3,
        min_score: float = 0.35,
    ) -> list[RetrievedVisual]:
        """Attach visual crops whose source text fuzzy-matches selected chunks."""
        if not retrieved_chunks or not self.visuals:
            return []

        matches: list[RetrievedVisual] = []
        seen: set[str] = set()
        for visual in self.visuals:
            best_score = 0.0
            for retrieved in retrieved_chunks:
                if visual.doc_id != retrieved.chunk.doc_id:
                    continue
                score = _visual_chunk_score(visual, retrieved.chunk)
                best_score = max(best_score, score)

            if best_score >= min_score and visual.evidence_id not in seen:
                matches.append(RetrievedVisual(visual, best_score))
                seen.add(visual.evidence_id)

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]


def _visual_chunk_score(visual: VisualEvidence, chunk: TextChunk) -> float:
    visual_text = _normalize_text(f"{visual.key_text} {visual.analysis_text}")
    chunk_text = _normalize_text(chunk.text)
    if not visual_text or not chunk_text:
        return 0.0

    page_bonus = 0.15 if visual.page_number == chunk.page_number else 0.0
    overlap = _token_overlap(visual_text, chunk_text)
    sequence = SequenceMatcher(None, visual_text[:1200], chunk_text[:1200]).ratio()
    return min(1.0, page_bonus + (0.75 * overlap) + (0.25 * sequence))


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in left.split() if len(token) > 2}
    right_tokens = {token for token in right.split() if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)
