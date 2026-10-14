from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import get_settings
from .models import GroundedAnswer, RetrievedChunk, RetrievedVisual, SectionInsight


class LocalGroundedGenerator:
    """Deterministic generator used until an LLM-backed generator is configured."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = (
            ChatOpenAI(model=self.settings.openai_text_model)
            if self.settings.openai_api_key and self.settings.enable_llm_generation
            else None
        )

    def answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        visuals: list[RetrievedVisual],
    ) -> GroundedAnswer:
        if not chunks:
            return GroundedAnswer(
                answer="I could not find enough relevant text evidence in the indexed documents to answer this question.",
                chunks=(),
                visuals=tuple(visuals),
            )

        if self.llm:
            prompt = self._answer_prompt(question, chunks)
            response = self.llm.invoke(prompt)
            return GroundedAnswer(answer=str(response.content), chunks=tuple(chunks), visuals=tuple(visuals))

        evidence_lines = []
        for item in chunks[:3]:
            snippet = item.chunk.text[:450].strip()
            citation = f"{item.chunk.doc_id}, page {item.chunk.page_number}"
            evidence_lines.append(f"- {snippet} [{citation}]")

        answer = (
            f"Grounded draft answer for: {question}\n\n"
            "Most relevant evidence:\n"
            + "\n".join(evidence_lines)
        )
        return GroundedAnswer(answer=answer, chunks=tuple(chunks), visuals=tuple(visuals))

    def _answer_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        text_evidence = "\n\n".join(
            f"Source {idx}: document={item.chunk.doc_id}, page={item.chunk.page_number}, score={item.score:.3f}\n{item.chunk.text}"
            for idx, item in enumerate(chunks, start=1)
        )
        return f"""
You are a model governance review assistant.
Answer only from the supplied source evidence.
Every material claim must include a page citation.
If evidence is weak or missing, say that clearly.

Question:
{question}

Text evidence:
{text_evidence}
"""

    def section_insight(
        self,
        section: str,
        chunks: list[RetrievedChunk],
        visuals: list[RetrievedVisual],
    ) -> SectionInsight:
        if not chunks:
            return SectionInsight(
                section=section,
                summary="No strong source evidence was found for this section.",
                chunks=(),
                visuals=tuple(visuals),
            )

        citations = ", ".join(f"page {item.chunk.page_number}" for item in chunks[:3])
        summary = f"{chunks[0].chunk.text[:600].strip()} Evidence: {citations}."
        return SectionInsight(section=section, summary=summary, chunks=tuple(chunks), visuals=tuple(visuals))
