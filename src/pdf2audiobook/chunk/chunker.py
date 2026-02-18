"""Sentence-boundary chunking using spaCy.

Splits cleaned text into TTS-ready chunks that respect sentence boundaries,
character limits, and preserve chapter/paragraph structure metadata.
"""

from __future__ import annotations

import threading
from pathlib import Path

from pdf2audiobook.checkpoint import Checkpoint
from pdf2audiobook.config import ChunkConfig
from pdf2audiobook.models import ChapterChunks, ChunkMeta
from pdf2audiobook.progress import Progress

# Lazy-loaded spaCy model
_nlp = None
_nlp_lock = threading.Lock()


def _get_nlp(model_name: str):
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(model_name)
        except OSError:
            from spacy.cli import download
            download(model_name)
            _nlp = spacy.load(model_name)
        # Only need the sentencizer, disable everything else for speed
        _nlp.select_pipes(enable=["senter", "parser"] if "parser" in _nlp.pipe_names else [])
    return _nlp


def chunk_chapters(
    cleaned_paths: list[Path],
    output_dir: Path,
    chapter_titles: list[str],
    config: ChunkConfig,
    checkpoint: Checkpoint,
    progress: Progress,
) -> list[Path]:
    """Chunk all cleaned chapter texts and write results to disk.

    Returns paths to chunk JSON files.
    """
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []

    nlp = _get_nlp(config.spacy_model)

    for ch_idx, cleaned_path in enumerate(cleaned_paths):
        out_path = chunks_dir / f"chapter_{ch_idx:03d}.json"
        result_paths.append(out_path)

        if checkpoint.is_done(ch_idx, "chunked"):
            progress.detail(f"Chapter {ch_idx} already chunked, skipping")
            continue

        title = chapter_titles[ch_idx] if ch_idx < len(chapter_titles) else f"Chapter {ch_idx + 1}"
        progress.chapter(ch_idx, len(cleaned_paths), f"Chunking: {title}")

        text = cleaned_path.read_text(encoding="utf-8")
        chunks = _chunk_text(text, ch_idx, config.max_chars, nlp)

        chapter_chunks = ChapterChunks(
            chapter_index=ch_idx,
            chapter_title=title,
            chunks=chunks,
        )
        chapter_chunks.save(out_path)
        checkpoint.mark(ch_idx, "chunked")
        progress.chapter_done(ch_idx, len(cleaned_paths))

    return result_paths


def chunk_one_chapter(
    ch_idx: int,
    cleaned_path: Path,
    output_dir: Path,
    title: str,
    config: ChunkConfig,
) -> Path:
    """Chunk a single cleaned chapter and write result to disk. Thread-safe.

    Returns the path to the chunk JSON file.
    """
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_path = chunks_dir / f"chapter_{ch_idx:03d}.json"

    with _nlp_lock:
        nlp = _get_nlp(config.spacy_model)

    text = cleaned_path.read_text(encoding="utf-8")

    with _nlp_lock:
        chunks = _chunk_text(text, ch_idx, config.max_chars, nlp)

    chapter_chunks = ChapterChunks(
        chapter_index=ch_idx,
        chapter_title=title,
        chunks=chunks,
    )
    chapter_chunks.save(out_path)
    return out_path


def _chunk_text(
    text: str,
    chapter_index: int,
    max_chars: int,
    nlp,
) -> list[ChunkMeta]:
    """Split text into chunks respecting sentence boundaries and char limits."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[ChunkMeta] = []
    sentence_index = 0

    for para_idx, paragraph in enumerate(paragraphs):
        sentences = _split_sentences(paragraph, nlp)

        current_chunk = ""
        chunk_start_sentence = sentence_index

        for sent_idx_in_para, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed the limit, flush current chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
                is_last_para = para_idx == len(paragraphs) - 1
                chunks.append(
                    ChunkMeta(
                        text=current_chunk.strip(),
                        chapter_index=chapter_index,
                        paragraph_index=para_idx,
                        sentence_index=chunk_start_sentence,
                        is_paragraph_end=False,
                        is_chapter_end=False,
                    )
                )
                current_chunk = ""
                chunk_start_sentence = sentence_index

            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            sentence_index += 1

        # Flush remaining text for this paragraph
        if current_chunk.strip():
            is_last_para = para_idx == len(paragraphs) - 1
            chunks.append(
                ChunkMeta(
                    text=current_chunk.strip(),
                    chapter_index=chapter_index,
                    paragraph_index=para_idx,
                    sentence_index=chunk_start_sentence,
                    is_paragraph_end=True,
                    is_chapter_end=is_last_para,
                )
            )
            current_chunk = ""

    # Mark the very last chunk as chapter end
    if chunks:
        chunks[-1].is_chapter_end = True

    return chunks


def _split_sentences(text: str, nlp) -> list[str]:
    """Use spaCy to split text into sentences."""
    # Process in chunks if text is very long (spaCy has a default 1M char limit)
    max_len = nlp.max_length
    if len(text) > max_len:
        sentences = []
        for i in range(0, len(text), max_len):
            doc = nlp(text[i : i + max_len])
            sentences.extend(sent.text for sent in doc.sents)
        return sentences

    doc = nlp(text)
    return [sent.text for sent in doc.sents]
