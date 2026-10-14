from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from io import BytesIO

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image

from .config import get_settings


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int
    reason: str

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


class VisualCropLocator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.enabled = bool(
            self.settings.openai_api_key
            and self.settings.enable_llm_visual_analysis
        )
        self.llm = ChatOpenAI(model=self.settings.openai_vision_model) if self.enabled else None

    def locate(self, page_image: Image.Image, page_text: str, visual_kind: str) -> CropBox | None:
        if not self.llm:
            return None

        data_url = _image_data_url(page_image)
        width, height = page_image.size
        prompt = f"""
You are locating visual evidence in a PDF page image.
Find the tight bounding box around the main {visual_kind} evidence on this page.
Prefer a table, chart, graph, figure, or exhibit that matches the nearby page text.

Return only JSON with this schema:
{{
  "left": 0.0,
  "top": 0.0,
  "right": 1.0,
  "bottom": 1.0,
  "reason": "short reason"
}}

Coordinates must be normalized from 0 to 1 relative to image size.
Do not include markdown.

Page image size: {width}x{height}
Nearby text:
{page_text[:1600]}
"""
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
        return _parse_crop_box(str(response.content), width, height)


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _parse_crop_box(content: str, width: int, height: int) -> CropBox | None:
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
        left = _clip(float(payload["left"]))
        top = _clip(float(payload["top"]))
        right = _clip(float(payload["right"]))
        bottom = _clip(float(payload["bottom"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if right <= left or bottom <= top:
        return None

    return CropBox(
        left=int(left * width),
        top=int(top * height),
        right=int(right * width),
        bottom=int(bottom * height),
        reason=str(payload.get("reason", "llm_selected_crop")),
    )


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
