from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .config import get_settings
from .models import VisualEvidence


class VisualAnalyzer:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.enabled = bool(
            self.settings.openai_api_key
            and self.settings.enable_llm_visual_analysis
        )
        self.llm = (
            ChatOpenAI(model=self.settings.openai_vision_model)
            if self.enabled
            else None
        )

    def enrich(self, visuals: list[VisualEvidence]) -> list[VisualEvidence]:
        if not self.llm:
            return visuals

        enriched: list[VisualEvidence] = []
        for visual in visuals:
            analysis = self._describe_visual(visual)
            enriched.append(
                VisualEvidence(
                    evidence_id=visual.evidence_id,
                    doc_id=visual.doc_id,
                    page_number=visual.page_number,
                    kind=visual.kind,
                    image_path=visual.image_path,
                    key_text=f"{visual.key_text} {analysis}".strip(),
                    analysis_text=analysis,
                    bbox=visual.bbox,
                )
            )
        return enriched

    def _describe_visual(self, visual: VisualEvidence) -> str:
        image_bytes = visual.image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{encoded}"
        prompt = (
            "Describe this governance-document visual for retrieval. "
            "Mention table titles, axes, risk/control labels, ratings, trends, and key numbers. "
            "Be concise and do not invent unreadable details."
        )
        response = self.llm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                )
            ]
        )
        return str(response.content)
