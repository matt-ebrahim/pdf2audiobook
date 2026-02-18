"""Pipeline orchestrator — runs all 4 stages with checkpoint/resume."""

from __future__ import annotations

from pathlib import Path

from pdf2audiobook.checkpoint import Checkpoint
from pdf2audiobook.config import Config
from pdf2audiobook.models import Chapter
from pdf2audiobook.progress import Progress


def run_pipeline(
    pdf_path: Path,
    output_dir: Path,
    config: Config,
) -> Path:
    """Run the full PDF-to-audiobook pipeline.

    Returns the path to the final output file/directory.
    """
    progress = Progress()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = Checkpoint.load_or_create(checkpoint_path, pdf_path)

    # ── Stage 1: Parse ──────────────────────────────────────
    progress.stage("1 — Parse PDF")

    from pdf2audiobook.parse import parse_pdf

    # Check if all chapters are already parsed
    if checkpoint.total_chapters > 0 and all(
        checkpoint.is_done(i, "parsed") for i in range(checkpoint.total_chapters)
    ):
        progress.detail("All chapters already parsed, loading from disk...")
        chapters = _load_parsed_chapters(output_dir, checkpoint.total_chapters)
    else:
        chapters = parse_pdf(pdf_path, output_dir, config.parse, checkpoint, progress)

    total = len(chapters)
    progress.done(f"Parsed {total} chapter(s)")

    for ch in chapters:
        progress.detail(
            f"  Ch {ch.index}: {ch.title} "
            f"(pages {ch.start_page}-{ch.end_page}, {len(ch.sections)} sections)"
        )

    # ── Stages 2-4: Streaming Clean → Chunk → Synthesize ───
    progress.stage("2-4 — Streaming pipeline (clean → chunk → synth)")

    # Auto-adjust chunk size for API TTS engines
    if config.synth.tts_engine in ("openai", "elevenlabs"):
        if config.chunk.max_chars == 400:  # still at default
            config.chunk.max_chars = 4096
            progress.detail(
                f"Auto-adjusted chunk size to {config.chunk.max_chars} for API TTS"
            )

    from pdf2audiobook.streaming import StreamingPipeline

    pipeline = StreamingPipeline(
        chapters=chapters,
        output_dir=output_dir,
        config=config,
        checkpoint=checkpoint,
        progress=progress,
    )
    results = pipeline.run()

    progress.done(f"Processed {len(results)} chapter(s) through streaming pipeline")

    # ── Final: M4B assembly ────────────────────────────────
    progress.stage("5 — Final assembly")

    import subprocess

    from pdf2audiobook.synth.stitcher import create_m4b

    audio_dir = output_dir / "audio"
    final_dir = output_dir / "final"

    chapter_mp3s = [audio_dir / f"chapter_{i:03d}.mp3" for i in range(total)]
    chapter_titles = [ch.title for ch in chapters]

    output_format = config.synth.output_format
    if output_format == "m4b":
        progress.detail("Creating M4B with chapter markers...")
        final_path = final_dir / "audiobook.m4b"
        try:
            create_m4b(chapter_mp3s, chapter_titles, final_path, config.synth, progress)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            progress.warn(f"M4B creation failed ({e}), falling back to MP3 output")
            output_format = "mp3"

    if output_format == "mp3":
        final_dir.mkdir(parents=True, exist_ok=True)
        progress.detail(f"Output: {len(chapter_mp3s)} MP3 files in {audio_dir}")
        final_path = audio_dir

    progress.stage("Complete!")
    progress.print_timing_summary()
    progress.done(f"Output: {final_path}")

    return final_path


def _load_parsed_chapters(output_dir: Path, total: int) -> list[Chapter]:
    """Load previously parsed chapters from disk."""
    chapters = []
    parsed_dir = output_dir / "parsed"
    for i in range(total):
        json_path = parsed_dir / f"chapter_{i:03d}.json"
        if json_path.exists():
            chapters.append(Chapter.from_json(json_path))
        else:
            # Fallback: create a minimal chapter from the markdown file
            md_path = parsed_dir / f"chapter_{i:03d}.md"
            if md_path.exists():
                from pdf2audiobook.models import Section
                text = md_path.read_text(encoding="utf-8")
                chapters.append(
                    Chapter(
                        index=i,
                        title=f"Chapter {i + 1}",
                        sections=[Section(type="body", content=text)],
                    )
                )
    return chapters
