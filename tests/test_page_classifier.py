from docgov.page_classifier import classify_page, infer_visual_kind


def test_classify_page_detects_visual_keyword() -> None:
    likely, reasons = classify_page("Figure 3 shows model monitoring drift across quarterly review windows.")
    assert likely
    assert "visual_keyword" in reasons


def test_classify_page_detects_dense_numeric_rows() -> None:
    text = "Q1 10 20 30 40\nQ2 11 21 31 41\nNarrative controls are reviewed annually."
    likely, reasons = classify_page(text)
    assert likely
    assert "dense_numeric_rows" in reasons


def test_infer_visual_kind_prefers_table() -> None:
    assert infer_visual_kind("Table 4: Model risk controls and owners") == "table"

