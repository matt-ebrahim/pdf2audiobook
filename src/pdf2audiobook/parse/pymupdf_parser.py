"""PyMuPDF-based parser for simple, text-native PDFs.

Processes page-by-page in a streaming fashion. Detects chapter boundaries
via font-size analysis and heading patterns.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from pdf2audiobook.checkpoint import Checkpoint
from pdf2audiobook.models import Chapter, Section
from pdf2audiobook.progress import Progress

# Patterns that indicate a chapter heading
CHAPTER_PATTERNS = [
    re.compile(r"^chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^chapter\s+[IVXLC]+", re.IGNORECASE),
    re.compile(r"^part\s+\d+", re.IGNORECASE),
    re.compile(r"^part\s+[IVXLC]+", re.IGNORECASE),
    re.compile(r"^section\s+\d+", re.IGNORECASE),
    re.compile(r"^\d+\.\s+\S"),  # "1. Introduction"
    re.compile(r"^prologue$", re.IGNORECASE),
    re.compile(r"^epilogue$", re.IGNORECASE),
    re.compile(r"^introduction$", re.IGNORECASE),
    re.compile(r"^conclusion$", re.IGNORECASE),
    re.compile(r"^appendix", re.IGNORECASE),
    re.compile(r"^references$", re.IGNORECASE),
    re.compile(r"^bibliography$", re.IGNORECASE),
    re.compile(r"^abstract$", re.IGNORECASE),
]


class PyMuPDFParser:
    """Page-by-page streaming parser using PyMuPDF."""

    def parse(
        self,
        pdf_path: Path,
        output_dir: Path,
        checkpoint: Checkpoint,
        progress: Progress,
    ) -> list[Chapter]:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # First pass: analyze font sizes to determine heading thresholds
        font_profile = self._profile_fonts(doc, sample_pages=min(20, total_pages))

        # Second pass: extract content page by page, building chapters
        all_sections: list[Section] = []
        for page_num in range(total_pages):
            page = doc[page_num]
            sections = self._extract_page(page, page_num, font_profile)
            all_sections.extend(sections)

        doc.close()

        # Group sections into chapters
        chapters = self._build_chapters(all_sections)

        if not chapters:
            # Fallback: treat entire document as one chapter
            chapters = [
                Chapter(
                    index=0,
                    title="Full Document",
                    sections=all_sections,
                    start_page=0,
                    end_page=total_pages - 1,
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

    def _profile_fonts(
        self, doc: fitz.Document, sample_pages: int
    ) -> dict:
        """Analyze font sizes across sample pages to determine body vs heading sizes."""
        size_counts: dict[float, int] = {}
        all_sizes: list[float] = []

        for page_num in range(min(sample_pages, len(doc))):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue
                        size = round(span["size"], 1)
                        char_count = len(text)
                        size_counts[size] = size_counts.get(size, 0) + char_count
                        all_sizes.append(size)

        if not size_counts:
            return {"body_size": 12.0, "h1_min": 18.0, "h2_min": 14.0}

        # Body size = most frequently used font size (by character count)
        body_size = max(size_counts, key=size_counts.get)

        return {
            "body_size": body_size,
            "h1_min": body_size * 1.5,  # 50%+ larger = H1
            "h2_min": body_size * 1.2,  # 20%+ larger = H2
        }

    def _extract_page(
        self, page: fitz.Page, page_num: int, font_profile: dict
    ) -> list[Section]:
        """Extract structured sections from a single page."""
        sections: list[Section] = []
        text_dict = page.get_text("dict")
        page_height = page.rect.height
        body_size = font_profile["body_size"]

        current_text: list[str] = []
        current_type = "body"
        current_level = 0

        for block in text_dict.get("blocks", []):
            if block.get("type") == 1:
                # Image block — mark as figure
                if current_text:
                    sections.append(
                        Section(
                            type=current_type,
                            content=" ".join(current_text).strip(),
                            level=current_level,
                            page=page_num,
                        )
                    )
                    current_text = []
                    current_type = "body"
                    current_level = 0
                sections.append(
                    Section(type="figure_caption", content="[Figure]", page=page_num)
                )
                continue

            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_text_parts: list[str] = []
                line_sizes: list[float] = []
                line_bold = False
                line_y = 0.0

                for span in line.get("spans", []):
                    text = span["text"]
                    if not text.strip():
                        continue
                    line_text_parts.append(text)
                    line_sizes.append(span["size"])
                    line_y = span["origin"][1]
                    if "bold" in span.get("font", "").lower():
                        line_bold = True

                line_text = "".join(line_text_parts).strip()
                if not line_text:
                    continue

                avg_size = sum(line_sizes) / len(line_sizes) if line_sizes else body_size

                # Determine line type
                is_footnote = avg_size < body_size * 0.85 and line_y > page_height * 0.8
                is_h1 = avg_size >= font_profile["h1_min"]
                is_h2 = not is_h1 and avg_size >= font_profile["h2_min"]
                is_heading = is_h1 or is_h2 or (
                    line_bold
                    and avg_size >= body_size
                    and len(line_text) < 120
                )

                if is_footnote:
                    line_type, level = "footnote", 0
                elif is_h1 or self._matches_chapter_pattern(line_text):
                    line_type, level = "heading", 1
                elif is_h2:
                    line_type, level = "heading", 2
                elif is_heading:
                    line_type, level = "heading", 2
                else:
                    line_type, level = "body", 0

                # If type changed, flush current buffer
                if line_type != current_type or (line_type == "heading"):
                    if current_text:
                        content = " ".join(current_text).strip()
                        if content:
                            sections.append(
                                Section(
                                    type=current_type,
                                    content=content,
                                    level=current_level,
                                    page=page_num,
                                )
                            )
                    current_text = [line_text]
                    current_type = line_type
                    current_level = level
                else:
                    current_text.append(line_text)

        # Flush remaining
        if current_text:
            content = " ".join(current_text).strip()
            if content:
                sections.append(
                    Section(
                        type=current_type,
                        content=content,
                        level=current_level,
                        page=page_num,
                    )
                )

        return sections

    def _matches_chapter_pattern(self, text: str) -> bool:
        """Check if text matches a chapter heading pattern."""
        text = text.strip()
        return any(p.match(text) for p in CHAPTER_PATTERNS)

    def _build_chapters(self, sections: list[Section]) -> list[Chapter]:
        """Group sections into chapters based on H1 headings."""
        chapters: list[Chapter] = []
        current_sections: list[Section] = []
        current_title = ""
        current_start_page = 0

        for section in sections:
            is_chapter_break = (
                section.type == "heading"
                and section.level == 1
            )

            if is_chapter_break and (current_sections or current_title):
                # Save the current chapter
                chapters.append(
                    Chapter(
                        index=len(chapters),
                        title=current_title or f"Chapter {len(chapters) + 1}",
                        sections=current_sections,
                        start_page=current_start_page,
                        end_page=current_sections[-1].page if current_sections else current_start_page,
                    )
                )
                current_sections = []

            if is_chapter_break:
                current_title = section.content
                current_start_page = section.page

            current_sections.append(section)

        # Don't forget the last chapter
        if current_sections:
            chapters.append(
                Chapter(
                    index=len(chapters),
                    title=current_title or f"Chapter {len(chapters) + 1}",
                    sections=current_sections,
                    start_page=current_start_page,
                    end_page=current_sections[-1].page if current_sections else current_start_page,
                )
            )

        return chapters
