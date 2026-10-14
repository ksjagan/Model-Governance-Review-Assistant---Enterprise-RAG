from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
import hashlib
from pathlib import Path

import pypdfium2 as pdfium
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from .config import get_settings
from .models import DocumentIndex, DocumentPage, TextChunk, VisualEvidence
from .page_classifier import classify_page, infer_visual_kind
from .visual_analysis import VisualAnalyzer
from .visual_cropper import VisualCropLocator


@dataclass(frozen=True)
class _ProcessedPage:
    page: DocumentPage
    chunks: list[TextChunk]
    visuals: list[VisualEvidence]


def parse_pdf(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int = 1200,
    overlap: int = 180,
    page_workers: int | None = None,
) -> DocumentIndex:
    doc_id = _doc_id(pdf_path)
    index = DocumentIndex(doc_id=doc_id)
    asset_dir = output_dir / doc_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    pdf_bytes = pdf_path.read_bytes()

    page_count = _page_count(pdf_bytes)
    workers = page_workers or get_settings().pdf_page_workers
    workers = max(1, min(workers, page_count or 1))

    process_page = partial(
        _process_page,
        pdf_bytes=pdf_bytes,
        doc_id=doc_id,
        asset_dir=asset_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        processed_pages = list(executor.map(process_page, range(page_count)))

    for processed in sorted(processed_pages, key=lambda item: item.page.page_number):
        index.pages.append(processed.page)
        index.chunks.extend(processed.chunks)
        index.visual_evidence.extend(processed.visuals)

    return index


def _doc_id(pdf_path: Path) -> str:
    digest = hashlib.sha1(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{pdf_path.stem}-{digest}"


def _page_count(pdf_bytes: bytes) -> int:
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        return len(pdf)
    finally:
        pdf.close()


def _process_page(
    page_idx: int,
    pdf_bytes: bytes,
    doc_id: str,
    asset_dir: Path,
    chunk_size: int,
    overlap: int,
) -> _ProcessedPage:
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        page_number = page_idx + 1
        page = pdf[page_idx]
        text = _extract_page_text(page)
        likely_visual, reasons = classify_page(text)
        document_page = DocumentPage(
            doc_id=doc_id,
            page_number=page_number,
            text=text,
            likely_visual=likely_visual,
            visual_reasons=reasons,
        )
        chunks = _chunk_page(doc_id, page_number, text, chunk_size, overlap)
        visuals: list[VisualEvidence] = []
        if likely_visual:
            crop_locator = VisualCropLocator()
            visual_analyzer = VisualAnalyzer()
            visuals = _extract_visual_crops(doc_id, page_number, page, text, asset_dir, crop_locator)
            visuals = visual_analyzer.enrich(visuals)
        return _ProcessedPage(page=document_page, chunks=chunks, visuals=visuals)
    finally:
        pdf.close()


def _extract_page_text(page: pdfium.PdfPage) -> str:
    text_page = page.get_textpage()
    try:
        return text_page.get_text_range() or ""
    finally:
        text_page.close()


def _chunk_page(doc_id: str, page_number: int, text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    clean = " ".join(text.split())
    if not clean:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(clean)
    return [
        TextChunk(
            chunk_id=f"{doc_id}:p{page_number}:c{idx}",
            doc_id=doc_id,
            page_number=page_number,
            text=chunk,
        )
        for idx, chunk in enumerate(chunks, start=1)
    ]


def _extract_visual_crops(
    doc_id: str,
    page_number: int,
    page: pdfium.PdfPage,
    page_text: str,
    asset_dir: Path,
    crop_locator: VisualCropLocator,
) -> list[VisualEvidence]:
    key_text = _nearby_key_text(page_text)
    kind = infer_visual_kind(key_text)
    rendered = _render_page(page, scale=2.0)
    llm_box = crop_locator.locate(rendered, key_text or page_text, kind)
    crop_boxes = [llm_box.as_tuple()] if llm_box else _candidate_crop_boxes(rendered.size, kind)

    evidence: list[VisualEvidence] = []
    for idx, box in enumerate(crop_boxes, start=1):
        crop = rendered.crop(box)
        image_path = asset_dir / f"p{page_number}_{kind}_crop_{idx}.png"
        crop.save(image_path)
        evidence.append(
            VisualEvidence(
                evidence_id=f"{doc_id}:p{page_number}:v{idx}",
                doc_id=doc_id,
                page_number=page_number,
                kind=kind,
                image_path=image_path,
                key_text=f"{key_text} {llm_box.reason if llm_box else 'heuristic_crop'}".strip(),
                bbox=tuple(float(value) for value in box),
            )
        )
    return evidence


def _render_page(page: pdfium.PdfPage, scale: float) -> Image.Image:
    bitmap = page.render(scale=scale)
    try:
        return bitmap.to_pil()
    finally:
        bitmap.close()


def _candidate_crop_boxes(size: tuple[int, int], kind: str) -> list[tuple[int, int, int, int]]:
    width, height = size
    margin_x = int(width * 0.06)
    margin_y = int(height * 0.08)

    if kind == "table":
        return [(margin_x, int(height * 0.18), width - margin_x, int(height * 0.88))]
    if kind in {"chart", "figure"}:
        return [(margin_x, int(height * 0.14), width - margin_x, int(height * 0.78))]
    return [(margin_x, margin_y, width - margin_x, height - margin_y)]


def _nearby_key_text(page_text: str, max_chars: int = 900) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    cue_lines = [
        line
        for line in lines
        if any(token in line.lower() for token in ("table", "figure", "fig.", "chart", "graph", "diagram"))
    ]
    selected = cue_lines[:6] or lines[:10]
    return " ".join(selected)[:max_chars]
