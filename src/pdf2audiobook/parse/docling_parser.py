"""Docling-based parser for complex PDFs (multi-column, tables, figures)."""

from __future__ import annotations

import re
from pathlib import Path

from pdf2audiobook.checkpoint import Checkpoint
from pdf2audiobook.models import Chapter, Section
from pdf2audiobook.progress import Progress


# Patterns for top-level chapter/section headings
_TOP_LEVEL_PATTERNS = [
    re.compile(r"^chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^chapter\s+[IVXLC]+", re.IGNORECASE),
    re.compile(r"^part\s+\d+", re.IGNORECASE),
    re.compile(r"^\d+\s+\S"),            # "1 Introduction" (no dot)
    re.compile(r"^\d+\.\s+\S"),          # "1. Introduction"
    re.compile(r"^abstract$", re.IGNORECASE),
    re.compile(r"^introduction$", re.IGNORECASE),
    re.compile(r"^conclusion[s]?$", re.IGNORECASE),
    re.compile(r"^discussion$", re.IGNORECASE),
    re.compile(r"^acknowledgements?$", re.IGNORECASE),
    re.compile(r"^references$", re.IGNORECASE),
    re.compile(r"^bibliography$", re.IGNORECASE),
    re.compile(r"^appendix", re.IGNORECASE),
    re.compile(r"^[A-Z]\s+\S"),          # "A Data sampling ratio"
]

# Sub-section patterns (should NOT trigger a chapter break)
_SUBSECTION_PATTERN = re.compile(r"^\d+\.\d+")  # "2.1", "2.1.3", etc.


class DoclingParser:
    """Parser using IBM Docling for complex layout analysis."""

    def parse(
        self,
        pdf_path: Path,
        output_dir: Path,
        checkpoint: Checkpoint,
        progress: Progress,
    ) -> list[Chapter]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            progress.error(
                "Docling not installed. Install with: pip install docling"
            )
            progress.detail("Falling back to PyMuPDF parser")
            from pdf2audiobook.parse.pymupdf_parser import PyMuPDFParser
            return PyMuPDFParser().parse(pdf_path, output_dir, checkpoint, progress)

        progress.detail("Running Docling document conversion (this may take a moment)...")

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        # Extract sections from the Docling document model
        sections = self._extract_sections(doc)

        # Group into chapters
        chapters = self._build_chapters(sections)

        if not chapters:
            chapters = [
                Chapter(
                    index=0,
                    title="Full Document",
                    sections=sections,
                    start_page=0,
                    end_page=0,
                )
            ]

        # Save each chapter to disk
        parsed_dir = output_dir / "parsed"
        for ch in chapters:
            if checkpoint.is_done(ch.index, "parsed"):
                progress.detail(f"Chapter {ch.index} already parsed, skipping")
                continue
            ch_path = parsed_dir / f"chapter_{ch.index:03d}.md"
            ch.save(ch_path)
            ch.save_json(parsed_dir / f"chapter_{ch.index:03d}.json")
            checkpoint.mark(ch.index, "parsed")
            progress.chapter_done(ch.index, len(chapters))

        return chapters

    def _extract_sections(self, doc) -> list[Section]:
        """Walk the Docling document tree and convert to our Section model."""
        sections: list[Section] = []

        for item, level in doc.iterate_items():
            label = getattr(item, "label", None) or type(item).__name__.lower()
            text = getattr(item, "text", "") or ""
            text = text.strip()

            if not text:
                continue

            label_lower = str(label).lower()

            if "section_header" in label_lower or "title" in label_lower:
                heading_level = self._classify_heading(text)
                sections.append(
                    Section(type="heading", content=text, level=heading_level)
                )
            elif "table" in label_lower:
                sections.append(Section(type="table", content=text))
            elif "picture" in label_lower or "figure" in label_lower:
                sections.append(
                    Section(type="figure_caption", content=text)
                )
            elif "caption" in label_lower:
                sections.append(
                    Section(type="figure_caption", content=text)
                )
            elif "footnote" in label_lower:
                sections.append(Section(type="footnote", content=text))
            elif "formula" in label_lower:
                sections.append(
                    Section(type="body", content=f"[equation: {text}]")
                )
            elif "list" in label_lower:
                sections.append(Section(type="body", content=text))
            else:
                sections.append(Section(type="body", content=text))

        return sections

    def _classify_heading(self, text: str) -> int:
        """Determine heading level from text content.

        Returns 1 for chapter/top-level sections, 2+ for sub-sections.
        This handles the common case where Docling reports all headings
        at the same depth level.
        """
        text = text.strip()

        # Sub-sections like "2.1 Pre-training data" → level 2
        if _SUBSECTION_PATTERN.match(text):
            # Count dots to determine depth: "2.1" = level 2, "2.1.3" = level 3
            num_part = text.split()[0] if text.split() else text
            depth = num_part.count(".") + 1
            return min(depth + 1, 6)  # "2.1" → level 2, "2.1.3" → level 3

        # Top-level patterns → level 1
        if any(p.match(text) for p in _TOP_LEVEL_PATTERNS):
            return 1

        # Default: treat as level 2 (sub-section)
        return 2

    def _build_chapters(self, sections: list[Section]) -> list[Chapter]:
        """Group sections into chapters based on level-1 headings.

        Title pages and author metadata before the first real section
        get grouped into a preamble chapter.
        """
        chapters: list[Chapter] = []
        current_sections: list[Section] = []
        current_title = ""
        preamble_done = False

        for section in sections:
            is_chapter_break = section.type == "heading" and section.level == 1

            if is_chapter_break:
                # Save accumulated sections as a chapter
                if current_sections:
                    # Skip preamble chapters that are just title/authors
                    if preamble_done or self._has_body_content(current_sections):
                        chapters.append(
                            Chapter(
                                index=len(chapters),
                                title=current_title or f"Section {len(chapters) + 1}",
                                sections=current_sections,
                            )
                        )
                    preamble_done = True
                current_sections = []
                current_title = section.content

            current_sections.append(section)

        # Don't forget the last chapter
        if current_sections:
            chapters.append(
                Chapter(
                    index=len(chapters),
                    title=current_title or f"Section {len(chapters) + 1}",
                    sections=current_sections,
                )
            )

        return chapters

    def _has_body_content(self, sections: list[Section]) -> bool:
        """Check if sections contain actual body text (not just headings/metadata)."""
        body_chars = sum(
            len(s.content) for s in sections if s.type == "body"
        )
        return body_chars > 50
