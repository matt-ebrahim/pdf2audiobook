"""LLM-based text cleaning for TTS preparation.

Uses LiteLLM as a unified interface to any LLM provider (OpenAI, Claude,
Gemini, Ollama, etc.). Cleans parsed chapter text by removing artifacts,
normalizing for speech, and stripping non-narrative content.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from pdf2audiobook.checkpoint import Checkpoint
from pdf2audiobook.config import CleanConfig, ApiKeys
from pdf2audiobook.models import Chapter
from pdf2audiobook.progress import Progress

SYSTEM_PROMPT = """\
You are a text preparation assistant for audiobook production. Your job is to \
clean and normalize extracted PDF text so it reads naturally when spoken aloud \
by a text-to-speech engine.

You MUST apply ALL of the following transformations:

## 1. Remove non-narrative junk
- Remove ALL headers, footers, page numbers, watermarks, and running titles.
- Remove arXiv identifiers, DOI strings, submission dates, and journal metadata \
  that appear as headings or floating text (e.g., "arXiv:2401.02385v2 [cs.CL] 4 Jun 2024").
- Remove author affiliations, email addresses, and institutional metadata.

## 2. Strip ALL citations and references
- Remove ALL parenthetical citations: (Smith et al., 2023), (Brown et al., 2020; \
  Chowdhery et al., 2022), (Touvron et al., 2023a,b).
- Remove ALL bracketed citations: [1], [2,3], [1-5], [Smith 2023].
- Remove superscript footnote markers (1, 2, *, †).
- If a sentence only makes sense with the citation, rephrase minimally to keep it \
  grammatical. For example: "as shown by Smith et al. (2023)" → "as shown in prior work".

## 3. Remove or replace tables
- Remove raw table data entirely (columns of numbers, benchmark scores, etc.).
- Replace each table with a single spoken sentence: \
  "Table N presents [brief description of what the table shows]." \
  If you cannot determine the table's purpose, use: "A table is omitted here."

## 4. Remove or replace figures
- Remove figure axis labels, tick marks, and data coordinates that leaked into text \
  (e.g., "0.00 0.25 0.50 0.75 1.00" or "1e12").
- Remove figure captions marked with [figure] or "Figure N:".
- Replace each figure reference with a brief spoken note: \
  "Figure N is omitted from this audio version." \
  Or if you can infer the content: "Figure N illustrates [brief description]."

## 5. Skip the references / bibliography section
- If you encounter a "References" or "Bibliography" section at the end, \
  remove it entirely. Do NOT read out reference entries.

## 6. Remove all URLs
- Remove all URLs (http://, https://, www., etc.) entirely.
- Remove DOI links.

## 7. Normalize for speech
- Expand abbreviations: "Fig." → "Figure", "Eq." → "Equation", "Tab." → "Table", \
  "Sec." → "Section", "vs." → "versus", "e.g." → "for example", \
  "i.e." → "that is", "et al." → "and colleagues", "etc." → "et cetera", \
  "approx." → "approximately", "w.r.t." → "with respect to".
- Normalize scientific notation: "4.0 × 10⁻⁴" → "four times ten to the negative four", \
  "1.1B" → "1.1 billion", "3T tokens" → "3 trillion tokens".
- Normalize currency: "$1.5M" → "1.5 million dollars".
- Normalize percentages: "50%" → "50 percent".
- Normalize common symbols: "β1" → "beta 1", "≥" → "greater than or equal to", \
  "→" → "leads to" or "becomes" (context-dependent).
- Fix broken hyphenation from line breaks: "com-\\nputer" → "computer".

## 8. Clean up structure
- Remove all markdown formatting (# headings, **, *, _, etc.).
- Output clean plain text with paragraph breaks as blank lines.
- Preserve the logical reading order and paragraph structure.
- Keep section headings as plain text (e.g., just "Pre-training" not "## 2.1 Pre-training").

## CRITICAL RULES
- Do NOT add any commentary, introductions, or meta-text.
- Do NOT summarize or shorten the text. Preserve ALL narrative content verbatim.
- Do NOT add transition phrases like "Moving on to..." or "Next, we discuss...".
- Return ONLY the cleaned text, nothing else.
- If the input is very short (just a title or heading), return it cleaned as-is."""

# Fallback regex patterns when LLM is unavailable
FALLBACK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\([\w\s]+et\s+al\.,?\s*\d{4}\w?(?:;\s*[\w\s]+et\s+al\.,?\s*\d{4}\w?)*\)"), ""),
    (re.compile(r"\[[\d,;\s\-]+\]"), ""),
    (re.compile(r"\[\w+(?:\s+et\s+al\.?)?,?\s*\d{4}\w?\]"), ""),
    (re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.MULTILINE), ""),
    (re.compile(r"(\w+)-\s*\n\s*(\w+)"), r"\1\2"),
    (re.compile(r"\*?\[figure\][^\n]*\*?", re.IGNORECASE), ""),
    (re.compile(r"<!--\s*table\s*-->.*?<!--\s*/table\s*-->", re.DOTALL), ""),
    (re.compile(r"\[Figure\]"), ""),
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    (re.compile(r"\bFig\.\s*"), "Figure "),
    (re.compile(r"\bEq\.\s*"), "Equation "),
    (re.compile(r"\bvs\.\s"), "versus "),
    (re.compile(r"\be\.g\.\s"), "for example "),
    (re.compile(r"\bi\.e\.\s"), "that is "),
    (re.compile(r"\bet\s+al\.\s"), "and colleagues "),
    (re.compile(r"\betc\.\s"), "et cetera "),
    (re.compile(r"\bapprox\.\s"), "approximately "),
    (re.compile(r"^>\s*\[footnote\]\s*", re.MULTILINE), ""),
    (re.compile(r"https?://\S+"), ""),
    (re.compile(r"\n{3,}"), "\n\n"),
]


# Module-level cost tracker (thread-safe)
_total_input_tokens = 0
_total_output_tokens = 0
_total_llm_calls = 0
_token_lock = threading.Lock()


def clean_chapters(
    chapters: list[Chapter],
    output_dir: Path,
    config: CleanConfig,
    api_keys: ApiKeys,
    checkpoint: Checkpoint,
    progress: Progress,
) -> list[Path]:
    """Clean all chapters and write results to disk.

    Returns paths to the cleaned text files.
    """
    global _total_input_tokens, _total_output_tokens, _total_llm_calls
    _total_input_tokens = 0
    _total_output_tokens = 0
    _total_llm_calls = 0

    cleaned_dir = output_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []

    use_llm = config.llm_backend != "none"
    if use_llm:
        _setup_api_keys(api_keys)

    # Filter out references section if configured
    if config.skip_references:
        skipped = sum(1 for ch in chapters if _is_references_section(ch.title))
        if skipped > 0:
            progress.detail(f"Skipping {skipped} reference/bibliography section(s)")

    total = len(chapters)
    for ch in chapters:
        out_path = cleaned_dir / f"chapter_{ch.index:03d}.txt"
        result_paths.append(out_path)

        if checkpoint.is_done(ch.index, "cleaned"):
            progress.detail(f"Chapter {ch.index} already cleaned, skipping")
            continue

        # Skip references sections — write empty file and mark done
        if config.skip_references and _is_references_section(ch.title):
            out_path.write_text("", encoding="utf-8")
            checkpoint.mark(ch.index, "cleaned")
            continue

        progress.chapter(ch.index, total, f"Cleaning: {ch.title}")

        raw_text = ch.to_markdown()

        if use_llm:
            try:
                cleaned = _llm_clean(raw_text, config.llm_model)
            except Exception as e:
                progress.error(f"LLM cleaning failed for chapter {ch.index}: {e}")
                progress.detail("Falling back to regex-based cleaning")
                cleaned = _regex_clean(raw_text)
        else:
            cleaned = _regex_clean(raw_text)

        out_path.write_text(cleaned.strip(), encoding="utf-8")
        checkpoint.mark(ch.index, "cleaned")
        progress.chapter_done(ch.index, total)

    # Report LLM usage
    if use_llm and _total_llm_calls > 0:
        progress.detail(
            f"LLM usage: {_total_llm_calls} calls, "
            f"{_total_input_tokens:,} input + {_total_output_tokens:,} output tokens"
        )
        # Estimate cost (gpt-4o-mini pricing as baseline: $0.15/1M input, $0.60/1M output)
        est_cost = (_total_input_tokens * 0.15 + _total_output_tokens * 0.60) / 1_000_000
        progress.detail(f"Estimated LLM cost: ${est_cost:.4f} (at gpt-4o-mini rates)")

    return result_paths


def clean_one_chapter(
    chapter: Chapter,
    output_dir: Path,
    config: CleanConfig,
    api_keys: ApiKeys,
) -> Path:
    """Clean a single chapter and write result to disk. Thread-safe.

    Returns the path to the cleaned text file.
    """
    cleaned_dir = output_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    out_path = cleaned_dir / f"chapter_{chapter.index:03d}.txt"

    use_llm = config.llm_backend != "none"
    if use_llm:
        _setup_api_keys(api_keys)

    # Skip references sections — write empty file
    if config.skip_references and _is_references_section(chapter.title):
        out_path.write_text("", encoding="utf-8")
        return out_path

    raw_text = chapter.to_markdown()

    if use_llm:
        try:
            cleaned = _llm_clean(raw_text, config.llm_model)
        except Exception:
            cleaned = _regex_clean(raw_text)
    else:
        cleaned = _regex_clean(raw_text)

    out_path.write_text(cleaned.strip(), encoding="utf-8")
    return out_path


def _setup_api_keys(api_keys: ApiKeys) -> None:
    """Set API keys as environment variables for LiteLLM."""
    import os
    if api_keys.openai:
        os.environ.setdefault("OPENAI_API_KEY", api_keys.openai)
    if api_keys.anthropic:
        os.environ.setdefault("ANTHROPIC_API_KEY", api_keys.anthropic)


def _llm_clean(text: str, model: str) -> str:
    """Clean text using LiteLLM (unified interface to any LLM provider)."""
    global _total_input_tokens, _total_output_tokens, _total_llm_calls
    import litellm

    litellm.suppress_debug_info = True

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
    )

    # Track token usage (thread-safe)
    usage = getattr(response, "usage", None)
    with _token_lock:
        if usage:
            _total_input_tokens += getattr(usage, "prompt_tokens", 0)
            _total_output_tokens += getattr(usage, "completion_tokens", 0)
        _total_llm_calls += 1

    return response.choices[0].message.content or text


def _is_references_section(title: str) -> bool:
    """Check if a chapter title indicates a references/bibliography section."""
    t = title.strip().lower()
    return t in (
        "references", "bibliography", "works cited", "literature cited",
        "reference", "ref", "refs",
    ) or t.startswith("references") or t.startswith("bibliography")


def _regex_clean(text: str) -> str:
    """Fallback regex-based cleaning when no LLM is available."""
    for pattern, replacement in FALLBACK_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()
