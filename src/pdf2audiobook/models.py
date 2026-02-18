"""Data models for the PDF-to-audiobook pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Section:
    """A structural unit within a parsed PDF page."""

    type: str  # heading, body, footnote, figure_caption, table
    content: str
    level: int = 0  # heading level 1-6, 0 for non-headings
    page: int = 0


@dataclass
class Chapter:
    """A chapter (or major section) extracted from a PDF."""

    index: int
    title: str
    sections: list[Section] = field(default_factory=list)
    start_page: int = 0
    end_page: int = 0

    def to_markdown(self) -> str:
        """Convert chapter to markdown string."""
        lines: list[str] = []
        for section in self.sections:
            if section.type == "heading":
                prefix = "#" * max(1, section.level)
                lines.append(f"{prefix} {section.content}\n")
            elif section.type == "body":
                lines.append(f"{section.content}\n")
            elif section.type == "footnote":
                lines.append(f"> [footnote] {section.content}\n")
            elif section.type == "figure_caption":
                lines.append(f"*[figure] {section.content}*\n")
            elif section.type == "table":
                lines.append(f"<!-- table -->\n{section.content}\n<!-- /table -->\n")
            else:
                lines.append(f"{section.content}\n")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save chapter markdown to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")

    def save_json(self, path: Path) -> None:
        """Save chapter as JSON for checkpoint purposes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> Chapter:
        """Load chapter from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        sections = [Section(**s) for s in data.pop("sections", [])]
        return cls(**data, sections=sections)


@dataclass
class ChunkMeta:
    """Metadata for a text chunk ready for TTS."""

    text: str
    chapter_index: int
    paragraph_index: int
    sentence_index: int
    is_paragraph_end: bool = False
    is_chapter_end: bool = False


@dataclass
class ChapterChunks:
    """All chunks for a single chapter."""

    chapter_index: int
    chapter_title: str
    chunks: list[ChunkMeta] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> ChapterChunks:
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = [ChunkMeta(**c) for c in data.pop("chunks", [])]
        return cls(**data, chunks=chunks)
