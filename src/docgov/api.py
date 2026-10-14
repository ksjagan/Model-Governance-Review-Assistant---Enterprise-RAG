from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import get_settings
from .pipeline import GovernanceReviewPipeline


settings = get_settings()
app = FastAPI(title="Model Governance Review Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.artifact_dir.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=str(settings.artifact_dir)), name="artifacts")

PIPELINES: dict[str, GovernanceReviewPipeline] = {}


class AskRequest(BaseModel):
    session_id: str
    question: str
    chunk_limit: int = 5
    visual_limit: int = 3
    visual_min_score: float | None = None


class ReviewRequest(BaseModel):
    session_id: str
    sections: list[str] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    session_id = uuid.uuid4().hex
    pipeline = GovernanceReviewPipeline(settings.artifact_dir)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_paths: list[Path] = []
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")
            path = Path(tmp) / file.filename
            with path.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            pdf_paths.append(path)

        indexes = pipeline.ingest(pdf_paths)

    PIPELINES[session_id] = pipeline
    return {
        "session_id": session_id,
        "stats": {
            "documents": len(indexes),
            "pages": sum(len(index.pages) for index in indexes),
            "chunks": sum(len(index.chunks) for index in indexes),
            "visuals": sum(len(index.visual_evidence) for index in indexes),
        },
        "visual_pages": [
            {
                "doc_id": page.doc_id,
                "page_number": page.page_number,
                "reasons": list(page.visual_reasons),
            }
            for index in indexes
            for page in index.pages
            if page.likely_visual
        ],
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    pipeline = _get_pipeline(request.session_id)
    answer = pipeline.ask(
        request.question,
        request.chunk_limit,
        request.visual_limit,
        request.visual_min_score,
    )
    return {
        "answer": answer.answer,
        "chunks": [
            {
                "chunk_id": item.chunk.chunk_id,
                "doc_id": item.chunk.doc_id,
                "page_number": item.chunk.page_number,
                "text": item.chunk.text,
                "score": item.score,
            }
            for item in answer.chunks
        ],
        "visuals": [_visual_payload(item) for item in answer.visuals],
    }


@app.post("/review")
def review(request: ReviewRequest) -> dict[str, Any]:
    pipeline = _get_pipeline(request.session_id)
    insights = pipeline.generate_review(request.sections)
    return {
        "sections": [
            {
                "section": insight.section,
                "summary": insight.summary,
                "chunks": [
                    {
                        "doc_id": item.chunk.doc_id,
                        "page_number": item.chunk.page_number,
                        "text": item.chunk.text,
                        "score": item.score,
                    }
                    for item in insight.chunks
                ],
                "visuals": [_visual_payload(item) for item in insight.visuals],
            }
            for insight in insights
        ]
    }


def _get_pipeline(session_id: str) -> GovernanceReviewPipeline:
    pipeline = PIPELINES.get(session_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload PDFs again.")
    return pipeline


def _visual_payload(item: Any) -> dict[str, Any]:
    evidence = item.evidence
    return {
        "evidence_id": evidence.evidence_id,
        "doc_id": evidence.doc_id,
        "page_number": evidence.page_number,
        "kind": evidence.kind,
        "key_text": evidence.key_text,
        "analysis_text": evidence.analysis_text,
        "image_url": f"/artifacts/{evidence.doc_id}/{evidence.image_path.name}",
        "score": item.score,
    }
