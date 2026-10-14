from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_text_model: str
    openai_vision_model: str
    enable_llm_generation: bool
    enable_llm_visual_analysis: bool
    fastapi_url: str
    artifact_dir: Path
    visual_match_min_score: float
    pdf_page_workers: int


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_text_model=os.getenv("OPENAI_TEXT_MODEL", "gpt-5"),
        openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-5"),
        enable_llm_generation=_env_bool("ENABLE_LLM_GENERATION", True),
        enable_llm_visual_analysis=_env_bool("ENABLE_LLM_VISUAL_ANALYSIS", True),
        fastapi_url=os.getenv("FASTAPI_URL", "http://127.0.0.1:8000"),
        artifact_dir=Path(os.getenv("ARTIFACT_DIR", "artifacts")),
        visual_match_min_score=_env_float("VISUAL_MATCH_MIN_SCORE", 0.35),
        pdf_page_workers=max(1, _env_int("PDF_PAGE_WORKERS", 10)),
    )
