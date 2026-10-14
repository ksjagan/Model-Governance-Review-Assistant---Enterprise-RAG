from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


VisualKind = Literal["table", "figure", "image", "chart", "unknown"]


@dataclass(frozen=True)
class DocumentPage:
    doc_id: str
    page_number: int
    text: str
    likely_visual: bool
    visual_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    page_number: int
    text: str


@dataclass(frozen=True)
class VisualEvidence:
    evidence_id: str
    doc_id: str
    page_number: int
    kind: VisualKind
    image_path: Path
    key_text: str
    analysis_text: str = ""
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class DocumentIndex:
    doc_id: str
    pages: list[DocumentPage] = field(default_factory=list)
    chunks: list[TextChunk] = field(default_factory=list)
    visual_evidence: list[VisualEvidence] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float


@dataclass(frozen=True)
class RetrievedVisual:
    evidence: VisualEvidence
    score: float


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    chunks: tuple[RetrievedChunk, ...]
    visuals: tuple[RetrievedVisual, ...]


@dataclass(frozen=True)
class SectionInsight:
    section: str
    summary: str
    chunks: tuple[RetrievedChunk, ...]
    visuals: tuple[RetrievedVisual, ...]
