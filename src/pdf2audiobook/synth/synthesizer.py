"""Synthesis orchestrator — generates audio for all chunks across all chapters.

Supports parallel chunk synthesis via multiprocessing (local TTS) or
ThreadPoolExecutor (API-based TTS).
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pdf2audiobook.checkpoint import Checkpoint
from pdf2audiobook.config import Config
from pdf2audiobook.models import ChapterChunks
from pdf2audiobook.progress import Progress
from pdf2audiobook.synth.base import TTSEngine
from pdf2audiobook.synth.stitcher import create_m4b, stitch_chapter


def get_tts_engine(config: Config) -> TTSEngine:
    """Create the configured TTS engine."""
    engine_name = config.synth.tts_engine

    if engine_name == "kokoro":
        from pdf2audiobook.synth.kokoro_tts import KokoroTTS
        return KokoroTTS(voice=config.synth.voice)
    elif engine_name == "openai":
        from pdf2audiobook.synth.openai_tts import OpenAITTS
        return OpenAITTS(
            voice=config.synth.voice,
            api_key=config.api_keys.openai,
        )
    elif engine_name == "elevenlabs":
        from pdf2audiobook.synth.elevenlabs_tts import ElevenLabsTTS
        return ElevenLabsTTS(
            voice=config.synth.voice,
            api_key=config.api_keys.elevenlabs,
        )
    elif engine_name == "chatterbox":
        from pdf2audiobook.synth.chatterbox_tts import ChatterboxTTS
        return ChatterboxTTS(voice=config.synth.voice)
    else:
        raise ValueError(f"Unknown TTS engine: {engine_name}")


def _get_worker_count(config: Config) -> int:
    """Determine the number of parallel workers for synthesis."""
    if config.synth.parallel_workers > 0:
        return config.synth.parallel_workers

    # Auto-detect: API engines can use many threads, local engines are CPU-bound
    if config.synth.tts_engine in ("openai", "elevenlabs"):
        return 8  # Network-bound, many concurrent requests
    else:
        # For local TTS, use CPU count but cap to avoid memory issues
        return min(os.cpu_count() or 1, 4)


def _synth_one_chunk(engine: TTSEngine, text: str, output_path: Path) -> tuple[Path, str]:
    """Synthesize a single chunk. Returns (path, error_or_empty)."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine.synthesize(text, output_path)
        return (output_path, "")
    except Exception as e:
        return (output_path, str(e))


def synthesize_one_chapter(
    ch_idx: int,
    chunk_path: Path,
    audio_dir: Path,
    config: Config,
    progress: Progress,
    engine: TTSEngine | None = None,
) -> Path:
    """Synthesize audio for a single chapter. Thread-safe (with external synth lock).

    Returns the path to the chapter MP3.
    """
    if engine is None:
        engine = get_tts_engine(config)
    workers = _get_worker_count(config)

    chapter_chunks = ChapterChunks.from_json(chunk_path)
    chapter_mp3 = audio_dir / f"chapter_{ch_idx:03d}.mp3"

    if not chapter_chunks.chunks:
        return chapter_mp3

    chunk_audio_dir = audio_dir / f"chapter_{ch_idx:03d}"

    # Build work items (skip already-generated chunks)
    work_items = []
    for i, chunk in enumerate(chapter_chunks.chunks):
        chunk_wav = chunk_audio_dir / f"chunk_{i:04d}.wav"
        if not chunk_wav.exists():
            work_items.append((i, chunk.text, chunk_wav))

    synth_errors = 0
    if not work_items:
        pass
    elif workers <= 1:
        for i, text, wav_path in work_items:
            path, err = _synth_one_chunk(engine, text, wav_path)
            if err:
                synth_errors += 1
                progress.error(f"TTS failed for chunk {i}: {err}")
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_synth_one_chunk, engine, text, wav_path): idx
                for idx, text, wav_path in work_items
            }
            for future in as_completed(futures):
                path, err = future.result()
                completed += 1
                if err:
                    synth_errors += 1
                    progress.error(f"TTS failed for {path.name}: {err}")

    # Check that at least some chunks succeeded
    if work_items and synth_errors == len(work_items):
        raise RuntimeError(
            f"All {synth_errors} chunks failed for chapter {ch_idx}"
        )

    # Stitch chunks into chapter MP3
    stitch_chapter(chunk_audio_dir, chapter_chunks, chapter_mp3, config.synth, progress)
    return chapter_mp3


def synthesize_all(
    chunk_paths: list[Path],
    output_dir: Path,
    config: Config,
    checkpoint: Checkpoint,
    progress: Progress,
) -> Path:
    """Synthesize audio for all chapters and produce the final audiobook."""
    engine = get_tts_engine(config)
    audio_dir = output_dir / "audio"
    final_dir = output_dir / "final"
    workers = _get_worker_count(config)

    progress.detail(f"Using {workers} parallel worker(s) for {engine.name} TTS")

    chapter_mp3s: list[Path] = []
    chapter_titles: list[str] = []

    for ch_idx, chunk_path in enumerate(chunk_paths):
        chapter_chunks = ChapterChunks.from_json(chunk_path)
        chapter_titles.append(chapter_chunks.chapter_title)
        chapter_mp3 = audio_dir / f"chapter_{ch_idx:03d}.mp3"
        chapter_mp3s.append(chapter_mp3)

        # Skip empty chapters (e.g., references that were stripped)
        if not chapter_chunks.chunks:
            checkpoint.mark(ch_idx, "synthesized")
            continue

        if checkpoint.is_done(ch_idx, "synthesized"):
            progress.detail(f"Chapter {ch_idx} already synthesized, skipping")
            continue

        progress.chapter(
            ch_idx, len(chunk_paths),
            f"Synthesizing: {chapter_chunks.chapter_title} ({len(chapter_chunks.chunks)} chunks)"
        )

        chunk_audio_dir = audio_dir / f"chapter_{ch_idx:03d}"

        # Build list of work items (skip already-generated chunks)
        work_items = []
        for i, chunk in enumerate(chapter_chunks.chunks):
            chunk_wav = chunk_audio_dir / f"chunk_{i:04d}.wav"
            if not chunk_wav.exists():
                work_items.append((i, chunk.text, chunk_wav))

        if not work_items:
            progress.detail("  All chunks already generated")
        elif workers <= 1:
            # Sequential synthesis
            for i, text, wav_path in work_items:
                path, err = _synth_one_chunk(engine, text, wav_path)
                if err:
                    progress.error(f"TTS failed for chunk {i}: {err}")
                if (i + 1) % 10 == 0:
                    progress.detail(
                        f"  Synthesized {i + 1}/{len(chapter_chunks.chunks)} chunks"
                    )
        else:
            # Parallel synthesis
            completed = 0
            errors = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_synth_one_chunk, engine, text, wav_path): idx
                    for idx, text, wav_path in work_items
                }
                for future in as_completed(futures):
                    path, err = future.result()
                    completed += 1
                    if err:
                        errors += 1
                        progress.error(f"TTS failed for {path.name}: {err}")
                    if completed % 10 == 0 or completed == len(work_items):
                        progress.detail(
                            f"  Synthesized {completed}/{len(work_items)} chunks"
                            + (f" ({errors} errors)" if errors else "")
                        )

        # Stitch chunks into chapter MP3
        progress.detail("  Stitching chapter audio...")
        stitch_chapter(
            chunk_audio_dir, chapter_chunks, chapter_mp3, config.synth, progress
        )

        checkpoint.mark(ch_idx, "synthesized")
        progress.chapter_done(ch_idx, len(chunk_paths))

    # Produce final output
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

    return final_path
