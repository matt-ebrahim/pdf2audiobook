"""Auto-detect PDF complexity by sampling the first N pages."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def detect_complexity(pdf_path: Path, sample_pages: int = 5) -> str:
    """Analyze the first N pages to determine if a PDF is 'complex' or 'simple'.

    Complex indicators:
    - Multi-column layouts (text blocks clustered in distinct x-regions)
    - Tables (grid-like line patterns)
    - Embedded images/figures
    - Footnote regions (small text at page bottom)

    Returns "complex" or "simple".
    """
    doc = fitz.open(pdf_path)
    pages_to_check = min(sample_pages, len(doc))

    signals = {
        "multi_column": 0,
        "has_images": 0,
        "has_tables": 0,
        "has_footnotes": 0,
    }

    for page_num in range(pages_to_check):
        page = doc[page_num]
        width = page.rect.width

        # --- Check for multi-column layout ---
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]  # type 0 = text

        if len(text_blocks) >= 4:
            # Collect x-midpoints of text blocks
            midpoints = [(b[0] + b[2]) / 2 for b in text_blocks]
            # Check if blocks cluster into left/right halves
            left = sum(1 for m in midpoints if m < width * 0.45)
            right = sum(1 for m in midpoints if m > width * 0.55)
            if left >= 2 and right >= 2:
                signals["multi_column"] += 1

        # --- Check for images ---
        images = page.get_images(full=True)
        if images:
            signals["has_images"] += 1

        # --- Check for table-like line patterns ---
        drawings = page.get_drawings()
        horiz_lines = 0
        vert_lines = 0
        for d in drawings:
            for item in d.get("items", []):
                if item[0] == "l":  # line
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 2:  # horizontal
                        horiz_lines += 1
                    elif abs(p1.x - p2.x) < 2:  # vertical
                        vert_lines += 1
        if horiz_lines >= 3 and vert_lines >= 2:
            signals["has_tables"] += 1

        # --- Check for footnotes (small text at bottom of page) ---
        text_dict = page.get_text("dict")
        page_height = page.rect.height
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    # Small text in the bottom 15% of the page
                    if span["size"] < 8 and span["origin"][1] > page_height * 0.85:
                        signals["has_footnotes"] += 1
                        break

    doc.close()

    # Decision: complex if any signal appears in >=40% of sampled pages,
    # or if images/tables appear at all
    threshold = max(1, pages_to_check * 0.4)
    is_complex = (
        signals["multi_column"] >= threshold
        or signals["has_images"] >= 2
        or signals["has_tables"] >= 1
    )

    return "complex" if is_complex else "simple"
