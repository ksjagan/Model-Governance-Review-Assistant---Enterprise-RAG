from __future__ import annotations

from pathlib import Path

from .config import get_settings
from .generation import LocalGroundedGenerator
from .models import DocumentIndex, GroundedAnswer, SectionInsight
from .pdf_parser import parse_pdf
from .retrieval import RetrievalStore


class GovernanceReviewPipeline:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.indexes: list[DocumentIndex] = []
        self.store: RetrievalStore | None = None
        self.generator = LocalGroundedGenerator()
        self.settings = get_settings()

    def ingest(self, pdf_paths: list[Path]) -> list[DocumentIndex]:
        self.indexes = [parse_pdf(path, self.artifact_dir) for path in pdf_paths]
        chunks = [chunk for index in self.indexes for chunk in index.chunks]
        visuals = [visual for index in self.indexes for visual in index.visual_evidence]
        self.store = RetrievalStore(chunks=chunks, visuals=visuals)
        return self.indexes

    def ask(
        self,
        question: str,
        chunk_limit: int = 5,
        visual_limit: int = 3,
        visual_min_score: float | None = None,
    ) -> GroundedAnswer:
        if self.store is None:
            raise RuntimeError("No documents have been ingested yet.")

        chunks = self.store.search_chunks(question, limit=chunk_limit)
        visuals = self.store.attach_visuals_to_chunks(
            chunks,
            limit=visual_limit,
            min_score=visual_min_score if visual_min_score is not None else self.settings.visual_match_min_score,
        )
        return self.generator.answer(question, chunks, visuals)

    def generate_review(self, sections: list[str] | None = None) -> list[SectionInsight]:
        if self.store is None:
            raise RuntimeError("No documents have been ingested yet.")

        review_sections = sections or [
            "model governance framework",
            "model risk controls",
            "validation and independent review",
            "monitoring metrics and thresholds",
            "issues, findings, and remediation",
            "approvals, ownership, and committees",
        ]

        insights: list[SectionInsight] = []
        for section in review_sections:
            chunks = self.store.search_chunks(section, limit=4)
            visuals = self.store.attach_visuals_to_chunks(
                chunks,
                limit=2,
                min_score=self.settings.visual_match_min_score,
            )
            insights.append(self.generator.section_insight(section, chunks, visuals))
        return insights
