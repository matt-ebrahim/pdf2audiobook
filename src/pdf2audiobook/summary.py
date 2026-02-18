"""Executive summary generation — produces a one-page overview of the document."""

from __future__ import annotations

import json
from pathlib import Path

from pdf2audiobook.checkpoint import Checkpoint
from pdf2audiobook.config import ApiKeys, CleanConfig
from pdf2audiobook.models import Chapter, Section
from pdf2audiobook.progress import Progress

SUMMARY_SYSTEM_PROMPT = """\
You are a world-class document analyst. Generate a comprehensive executive \
summary of the following document.

Guidelines:
- Length: 400-600 words (approximately one printed page)
- Begin with a clear statement of the document's purpose and main contribution
- Cover key findings, methodology (if applicable), and conclusions
- Maintain strict factual accuracy — include only information from the document
- Write in flowing prose with natural paragraph breaks
- Use clear, professional language suitable for audio narration
- Do not use bullet points, numbered lists, tables, or special formatting
- Do not add commentary, opinions, or information not present in the document

Return only the summary text, nothing else."""


def generate_and_inject_summary(
    chapters: list[Chapter],
    output_dir: Path,
    config: CleanConfig,
    api_keys: ApiKeys,
    checkpoint: Checkpoint,
    progress: Progress,
) -> list[Chapter]:
    """Generate an executive summary and inject it as the first chapter.

    Returns the updated chapters list with summary at index 0.
    Skips if summary was already generated (detected by first chapter title).
    """
    # Resume check — if first chapter is already the summary, skip
    if chapters and chapters[0].title == "Executive Summary":
        progress.detail("Executive summary already exists, skipping")
        return chapters

    progress.detail("Generating executive summary...")

    summary_text = _generate_summary_text(chapters, config, api_keys)
    if not summary_text.strip():
        progress.warn("Summary generation produced empty text, skipping")
        return chapters

    # Shift all existing files and checkpoint entries by +1
    old_count = len(chapters)
    _shift_chapter_files(output_dir, old_count)
    _shift_checkpoint_entries(checkpoint, old_count)

    # Reindex chapter objects
    for ch in chapters:
        ch.index += 1

    # Update index field inside parsed JSON files on disk
    _update_parsed_json_indices(output_dir / "parsed", old_count)

    # Create and save the summary chapter
    summary_ch = Chapter(
        index=0,
        title="Executive Summary",
        sections=[Section(type="body", content=summary_text)],
    )
    parsed_dir = output_dir / "parsed"
    summary_ch.save(parsed_dir / "chapter_000.md")
    summary_ch.save_json(parsed_dir / "chapter_000.json")

    # Write cleaned text directly (summary is already clean from the LLM)
    cleaned_dir = output_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    (cleaned_dir / "chapter_000.txt").write_text(summary_text, encoding="utf-8")

    # Update checkpoint
    checkpoint.mark(0, "parsed")
    checkpoint.mark(0, "cleaned")
    checkpoint.total_chapters = old_count + 1
    checkpoint.save()

    progress.done("Executive summary generated")
    return [summary_ch] + chapters


# ── LLM / Fallback Summary Generation ───────────────────────


def _generate_summary_text(
    chapters: list[Chapter],
    config: CleanConfig,
    api_keys: ApiKeys,
) -> str:
    """Generate summary text using LLM or extractive fallback."""
    full_text = "\n\n---\n\n".join(
        f"## {ch.title}\n\n{ch.to_markdown()}" for ch in chapters
    )

    # Truncate for LLM context limits (~25k tokens)
    max_chars = 100_000
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[Remaining text omitted for length...]"

    if config.llm_backend == "none":
        return _extractive_fallback(chapters)

    import os

    if api_keys.openai:
        os.environ.setdefault("OPENAI_API_KEY", api_keys.openai)
    if api_keys.anthropic:
        os.environ.setdefault("ANTHROPIC_API_KEY", api_keys.anthropic)

    try:
        import litellm

        litellm.suppress_debug_info = True

        response = litellm.completion(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": full_text},
            ],
            temperature=0.3,
        )
        result = response.choices[0].message.content
        return result.strip() if result else _extractive_fallback(chapters)
    except Exception:
        return _extractive_fallback(chapters)


def _extractive_fallback(chapters: list[Chapter]) -> str:
    """Simple extractive summary when LLM is unavailable."""
    parts = []
    for ch in chapters:
        text = ch.to_markdown().strip()
        if not text:
            continue
        first_para = text.split("\n\n")[0].strip()
        if first_para and len(first_para) > 30:
            parts.append(first_para)
        if len(parts) >= 5:
            break
    return "\n\n".join(parts) if parts else "Executive summary could not be generated."


# ── File & Checkpoint Reindexing ─────────────────────────────


def _shift_chapter_files(output_dir: Path, count: int) -> None:
    """Rename all chapter files to shift indices by +1."""
    dirs_and_exts = [
        (output_dir / "parsed", [".md", ".json"]),
        (output_dir / "cleaned", [".txt"]),
        (output_dir / "chunks", [".json"]),
    ]

    for d, exts in dirs_and_exts:
        if not d.exists():
            continue
        for i in range(count - 1, -1, -1):
            for ext in exts:
                src = d / f"chapter_{i:03d}{ext}"
                dst = d / f"chapter_{i + 1:03d}{ext}"
                if src.exists():
                    src.rename(dst)

    # Audio directory: MP3 files and chunk subdirectories
    audio = output_dir / "audio"
    if audio.exists():
        for i in range(count - 1, -1, -1):
            for name in (f"chapter_{i:03d}.mp3", f"chapter_{i:03d}"):
                src = audio / name
                dst = audio / name.replace(f"_{i:03d}", f"_{i + 1:03d}")
                if src.exists():
                    src.rename(dst)


def _shift_checkpoint_entries(checkpoint: Checkpoint, count: int) -> None:
    """Shift checkpoint chapter entries by +1."""
    for i in range(count - 1, -1, -1):
        old_key = str(i)
        new_key = str(i + 1)
        if old_key in checkpoint.chapters:
            checkpoint.chapters[new_key] = checkpoint.chapters.pop(old_key)


def _update_parsed_json_indices(parsed_dir: Path, old_count: int) -> None:
    """Update the 'index' field inside renamed parsed JSON files."""
    for i in range(1, old_count + 1):
        json_path = parsed_dir / f"chapter_{i:03d}.json"
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            data["index"] = i
            json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
