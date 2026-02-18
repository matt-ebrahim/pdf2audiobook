"""Checkpoint/resume system — tracks per-chapter stage completion."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = ("parsed", "cleaned", "chunked", "synthesized")


@dataclass
class ChapterStatus:
    parsed: bool = False
    cleaned: bool = False
    chunked: bool = False
    synthesized: bool = False


@dataclass
class Checkpoint:
    pdf_path: str = ""
    pdf_hash: str = ""
    parser_used: str = ""
    total_chapters: int = 0
    chapters: dict[str, ChapterStatus] = field(default_factory=dict)

    _path: Path | None = field(default=None, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def mark(self, chapter_index: int, stage: str) -> None:
        """Mark a stage as complete for a chapter and persist."""
        with self._lock:
            key = str(chapter_index)
            if key not in self.chapters:
                self.chapters[key] = ChapterStatus()
            setattr(self.chapters[key], stage, True)
            self.save()

    def is_done(self, chapter_index: int, stage: str) -> bool:
        key = str(chapter_index)
        if key not in self.chapters:
            return False
        return getattr(self.chapters[key], stage, False)

    def save(self) -> None:
        if self._path is None:
            return
        data = {
            "pdf_path": self.pdf_path,
            "pdf_hash": self.pdf_hash,
            "parser_used": self.parser_used,
            "total_chapters": self.total_chapters,
            "chapters": {k: asdict(v) for k, v in self.chapters.items()},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load_or_create(cls, checkpoint_path: Path, pdf_path: Path) -> Checkpoint:
        """Load an existing checkpoint or create a new one."""
        pdf_hash = _hash_file(pdf_path)

        if checkpoint_path.exists():
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if data.get("pdf_hash") == pdf_hash:
                chapters = {
                    k: ChapterStatus(**v) for k, v in data.get("chapters", {}).items()
                }
                cp = cls(
                    pdf_path=str(pdf_path),
                    pdf_hash=pdf_hash,
                    parser_used=data.get("parser_used", ""),
                    total_chapters=data.get("total_chapters", 0),
                    chapters=chapters,
                )
                cp._path = checkpoint_path
                return cp

        cp = cls(pdf_path=str(pdf_path), pdf_hash=pdf_hash)
        cp._path = checkpoint_path
        return cp


def _hash_file(path: Path) -> str:
    """Compute SHA-256 of a file's first 1MB (fast for large PDFs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]
