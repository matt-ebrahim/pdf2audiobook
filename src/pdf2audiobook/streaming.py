"""Streaming pipeline — processes chapters through clean→chunk→synth concurrently.

While one thread synthesizes audio (CPU-heavy), another cleans the next chapter
via LLM (I/O-heavy), achieving significant overlap and earlier audio availability.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pdf2audiobook.checkpoint import Checkpoint
from pdf2audiobook.config import Config
from pdf2audiobook.models import Chapter
from pdf2audiobook.progress import Progress


@dataclass
class ChapterResult:
    """Result of processing a single chapter through the streaming pipeline."""

    index: int
    cleaned_path: Path | None = None
    chunk_path: Path | None = None
    audio_path: Path | None = None
    error: str | None = None


class StreamingPipeline:
    """Processes chapters through clean→chunk→synth with concurrent overlap.

    Uses a thread pool to process multiple chapters simultaneously. The key win
    is overlapping LLM cleaning (I/O-bound) with TTS synthesis (CPU-bound).
    """

    def __init__(
        self,
        chapters: list[Chapter],
        output_dir: Path,
        config: Config,
        checkpoint: Checkpoint,
        progress: Progress,
        max_concurrent_chapters: int = 2,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._chapters = chapters
        self._output_dir = output_dir
        self._config = config
        self._checkpoint = checkpoint
        self._progress = progress
        self._max_concurrent = max_concurrent_chapters
        self._on_event = on_event
        self._synth_lock = threading.Lock()
        self._results: dict[int, ChapterResult] = {}

        # Create TTS engine once (shared across threads, serialized by _synth_lock)
        from pdf2audiobook.synth.synthesizer import get_tts_engine

        self._engine = get_tts_engine(config)

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(event, data)

    def run(self) -> dict[int, ChapterResult]:
        """Run the streaming pipeline. Returns per-chapter results."""
        total = len(self._chapters)
        self._progress.detail(
            f"Streaming pipeline: {total} chapters, "
            f"{self._max_concurrent} concurrent workers"
        )

        with ThreadPoolExecutor(max_workers=self._max_concurrent) as pool:
            futures = {
                pool.submit(self._process_chapter, ch): ch.index
                for ch in self._chapters
            }
            for future in as_completed(futures):
                ch_idx = futures[future]
                try:
                    result = future.result()
                    self._results[ch_idx] = result
                except Exception as e:
                    self._results[ch_idx] = ChapterResult(index=ch_idx, error=str(e))
                    self._progress.error(f"Chapter {ch_idx} failed: {e}")
                    self._emit("chapter_error", {"index": ch_idx, "error": str(e)})

        self._emit("pipeline_complete", {"total": total})
        return self._results

    def _process_chapter(self, chapter: Chapter) -> ChapterResult:
        """Process a single chapter: clean → chunk → synth."""
        ch_idx = chapter.index
        total = len(self._chapters)
        result = ChapterResult(index=ch_idx)

        # ── Clean ──
        self._emit("chapter_stage", {"index": ch_idx, "stage": "cleaning"})
        if not self._checkpoint.is_done(ch_idx, "cleaned"):
            self._progress.chapter(ch_idx, total, f"Cleaning: {chapter.title}")
            from pdf2audiobook.clean.cleaner import clean_one_chapter

            cleaned_path = clean_one_chapter(
                chapter, self._output_dir, self._config.clean, self._config.api_keys
            )
            self._checkpoint.mark(ch_idx, "cleaned")
        else:
            cleaned_path = self._output_dir / "cleaned" / f"chapter_{ch_idx:03d}.txt"
        result.cleaned_path = cleaned_path
        self._emit("chapter_text_ready", {"index": ch_idx})

        # ── Chunk ──
        self._emit("chapter_stage", {"index": ch_idx, "stage": "chunking"})
        if not self._checkpoint.is_done(ch_idx, "chunked"):
            self._progress.chapter(ch_idx, total, f"Chunking: {chapter.title}")
            from pdf2audiobook.chunk.chunker import chunk_one_chapter

            chunk_path = chunk_one_chapter(
                ch_idx, cleaned_path, self._output_dir, chapter.title, self._config.chunk
            )
            self._checkpoint.mark(ch_idx, "chunked")
        else:
            chunk_path = self._output_dir / "chunks" / f"chapter_{ch_idx:03d}.json"
        result.chunk_path = chunk_path

        # ── Synthesize (serialized — TTS engines aren't thread-safe) ──
        self._emit("chapter_stage", {"index": ch_idx, "stage": "synthesizing"})
        if not self._checkpoint.is_done(ch_idx, "synthesized"):
            self._progress.chapter(ch_idx, total, f"Synthesizing: {chapter.title}")
            from pdf2audiobook.synth.synthesizer import synthesize_one_chapter

            audio_dir = self._output_dir / "audio"
            with self._synth_lock:
                audio_path = synthesize_one_chapter(
                    ch_idx, chunk_path, audio_dir, self._config, self._progress,
                    engine=self._engine,
                )
            self._checkpoint.mark(ch_idx, "synthesized")
        else:
            audio_path = self._output_dir / "audio" / f"chapter_{ch_idx:03d}.mp3"
        result.audio_path = audio_path

        self._progress.done(f"Chapter {ch_idx} audio ready: {chapter.title}")
        self._emit("chapter_audio_ready", {"index": ch_idx, "title": chapter.title})
        return result
