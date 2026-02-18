"""PDF parsing stage — auto-detection and parser dispatch."""

from __future__ import annotations

from pathlib import Path

from pdf2audiobook.checkpoint import Checkpoint
from pdf2audiobook.config import ParseConfig
from pdf2audiobook.models import Chapter
from pdf2audiobook.progress import Progress


def parse_pdf(
    pdf_path: Path,
    output_dir: Path,
    config: ParseConfig,
    checkpoint: Checkpoint,
    progress: Progress,
) -> list[Chapter]:
    """Parse a PDF into chapters, auto-detecting the best parser.

    Returns a list of Chapter objects. Each chapter's markdown is also
    written to output_dir/parsed/chapter_NNN.md.
    """
    from pdf2audiobook.parse.detector import detect_complexity

    parser_choice = config.parser
    if parser_choice == "auto":
        complexity = detect_complexity(pdf_path, config.sample_pages)
        parser_choice = "docling" if complexity == "complex" else "pymupdf"
        progress.detail(f"Auto-detected PDF as {complexity} → using {parser_choice}")

    checkpoint.parser_used = parser_choice

    if parser_choice == "docling":
        from pdf2audiobook.parse.docling_parser import DoclingParser
        parser = DoclingParser()
    else:
        from pdf2audiobook.parse.pymupdf_parser import PyMuPDFParser
        parser = PyMuPDFParser()

    chapters = parser.parse(pdf_path, output_dir, checkpoint, progress)
    checkpoint.total_chapters = len(chapters)
    checkpoint.save()
    return chapters
