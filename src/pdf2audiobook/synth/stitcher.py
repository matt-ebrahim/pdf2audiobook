"""Audio stitching — concatenate chunks with pauses, normalize loudness, output M4B/MP3."""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

from pydub import AudioSegment

from pdf2audiobook.config import SynthConfig
from pdf2audiobook.models import ChapterChunks
from pdf2audiobook.progress import Progress


def stitch_chapter(
    chunk_audio_dir: Path,
    chapter_chunks: ChapterChunks,
    output_path: Path,
    config: SynthConfig,
    progress: Progress,
) -> Path:
    """Stitch all chunk audio files for a chapter into a single file.

    Inserts pauses between sentences, paragraphs, and at chapter boundaries.
    Normalizes loudness to target LUFS.
    """
    silence_sentence = AudioSegment.silent(duration=config.pause_sentence_ms)
    silence_paragraph = AudioSegment.silent(duration=config.pause_paragraph_ms)

    combined = AudioSegment.empty()
    prev_para_idx = -1

    for i, chunk in enumerate(chapter_chunks.chunks):
        chunk_path = chunk_audio_dir / f"chunk_{i:04d}.wav"
        if not chunk_path.exists():
            progress.warn(f"Missing audio file: {chunk_path}")
            continue

        audio = AudioSegment.from_file(str(chunk_path))

        # Trim trailing silence from chunk
        audio = _trim_trailing_silence(audio)

        if len(combined) > 0:
            # Insert pause based on boundary type
            if chunk.paragraph_index != prev_para_idx:
                combined += silence_paragraph
            else:
                combined += silence_sentence

        combined += audio
        prev_para_idx = chunk.paragraph_index

    if len(combined) == 0:
        progress.warn(f"No audio produced for chapter {chapter_chunks.chapter_index}")
        return output_path

    # Normalize loudness
    combined = _normalize_loudness(combined, config.target_lufs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), format="mp3", bitrate="192k")
    return output_path


def create_m4b(
    chapter_mp3s: list[Path],
    chapter_titles: list[str],
    output_path: Path,
    config: SynthConfig,
    progress: Progress,
) -> Path:
    """Combine chapter MP3s into a single M4B with chapter markers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not chapter_mp3s:
        progress.error("No chapter audio files to combine")
        return output_path

    # Build a concat file for ffmpeg
    concat_file = output_path.parent / "concat.txt"
    existing_mp3s = [p for p in chapter_mp3s if p.exists() and p.stat().st_size > 0]

    if not existing_mp3s:
        progress.error("No valid chapter audio files found")
        return output_path

    with open(concat_file, "w") as f:
        for mp3_path in existing_mp3s:
            f.write(f"file '{mp3_path.resolve()}'\n")

    # First, concatenate all MP3s into a single file
    combined_mp3 = output_path.parent / "combined.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(combined_mp3),
        ],
        capture_output=True,
        check=True,
    )

    # Build chapter metadata file
    metadata_file = output_path.parent / "chapters.txt"
    _write_chapter_metadata(existing_mp3s, chapter_titles, metadata_file)

    # Convert to M4B with chapter markers
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(combined_mp3),
            "-i", str(metadata_file),
            "-map", "0:a",
            "-map_metadata", "1",
            "-c:a", "aac", "-b:a", "128k",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )

    # Cleanup temp files
    concat_file.unlink(missing_ok=True)
    combined_mp3.unlink(missing_ok=True)
    metadata_file.unlink(missing_ok=True)

    return output_path


def _write_chapter_metadata(
    mp3_paths: list[Path], titles: list[str], metadata_path: Path
) -> None:
    """Write ffmpeg metadata file with chapter markers."""
    # Get durations of each chapter MP3
    durations_ms: list[int] = []
    for mp3_path in mp3_paths:
        audio = AudioSegment.from_file(str(mp3_path))
        durations_ms.append(len(audio))

    with open(metadata_path, "w") as f:
        f.write(";FFMETADATA1\n")
        start_ms = 0
        for i, duration in enumerate(durations_ms):
            title = titles[i] if i < len(titles) else f"Chapter {i + 1}"
            end_ms = start_ms + duration
            f.write("\n[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start_ms}\n")
            f.write(f"END={end_ms}\n")
            f.write(f"title={title}\n")
            start_ms = end_ms


def _trim_trailing_silence(audio: AudioSegment, threshold: int = -45) -> AudioSegment:
    """Trim silence from the end of an audio segment."""
    if len(audio) == 0:
        return audio

    # Work backwards in 10ms chunks
    chunk_size = 10
    end = len(audio)
    while end > chunk_size:
        chunk = audio[end - chunk_size : end]
        if chunk.dBFS > threshold:
            break
        end -= chunk_size

    return audio[:end] if end > 0 else audio


def _normalize_loudness(audio: AudioSegment, target_lufs: int = -16) -> AudioSegment:
    """Normalize audio loudness. Approximation using dBFS as a proxy for LUFS."""
    if len(audio) == 0 or audio.dBFS == float("-inf"):
        return audio

    # dBFS is a rough proxy for LUFS (integrated loudness is more complex,
    # but for speech this approximation works reasonably well)
    current_dbfs = audio.dBFS
    target_dbfs = target_lufs
    change = target_dbfs - current_dbfs

    # Clamp adjustment to avoid clipping
    change = min(change, 20)
    change = max(change, -20)

    return audio.apply_gain(change)
