from __future__ import annotations

import re


VISUAL_TERMS = re.compile(
    r"\b(table|figure|fig\.?|chart|graph|diagram|exhibit|appendix|matrix|heatmap)\b",
    re.IGNORECASE,
)

TABLE_HEADER = re.compile(
    r"\b(control|risk|rating|owner|status|threshold|metric|policy|standard|model|finding)\b",
    re.IGNORECASE,
)

NUMERIC_ROW = re.compile(r"(?:\b\d+(?:\.\d+)?%?\b\D*){4,}")


def classify_page(text: str) -> tuple[bool, tuple[str, ...]]:
    """Return whether a page should be inspected for visual/table evidence."""
    reasons: list[str] = []
    normalized = " ".join(text.split())

    if VISUAL_TERMS.search(normalized):
        reasons.append("visual_keyword")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numeric_rows = sum(1 for line in lines if NUMERIC_ROW.search(line))
    if numeric_rows >= 2:
        reasons.append("dense_numeric_rows")

    table_like_lines = 0
    for line in lines:
        has_header_term = TABLE_HEADER.search(line)
        has_column_spacing = bool(re.search(r"\S\s{2,}\S", line))
        has_pipe_grid = "|" in line
        if has_header_term and (has_column_spacing or has_pipe_grid):
            table_like_lines += 1

    if table_like_lines >= 2:
        reasons.append("table_like_layout")

    return bool(reasons), tuple(reasons)


def infer_visual_kind(context: str) -> str:
    lowered = context.lower()
    if "table" in lowered or "matrix" in lowered:
        return "table"
    if "chart" in lowered or "graph" in lowered or "heatmap" in lowered:
        return "chart"
    if "figure" in lowered or "fig." in lowered or "diagram" in lowered:
        return "figure"
    if "image" in lowered:
        return "image"
    return "unknown"

