"""Command-line interface for pdf2audiobook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pdf2audiobook",
        description="Convert PDFs to audiobooks via a 4-stage pipeline",
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the input PDF file",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./output/<pdf_name>)",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to TOML config file",
    )

    # Shortcut overrides (so you don't need a config file for simple use)
    parser.add_argument(
        "--tts",
        choices=["kokoro", "chatterbox", "openai", "elevenlabs"],
        default=None,
        help="TTS engine to use (overrides config)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help='LiteLLM model for cleaning (e.g. "gpt-4o-mini", "claude-sonnet-4-20250514"), or "none" to skip',
    )
    parser.add_argument(
        "--parser",
        choices=["auto", "docling", "pymupdf"],
        default=None,
        help="PDF parser to use (overrides config)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice name for TTS engine (overrides config)",
    )
    parser.add_argument(
        "--format",
        choices=["m4b", "mp3"],
        default=None,
        help="Output format (overrides config)",
    )

    args = parser.parse_args(argv)

    # Validate input
    if not args.pdf.exists():
        print(f"Error: PDF file not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    # Load config
    from pdf2audiobook.config import load_config

    config = load_config(args.config)

    # Apply CLI overrides
    if args.tts:
        config.synth.tts_engine = args.tts
    if args.llm:
        if args.llm == "none":
            config.clean.llm_backend = "none"
        else:
            config.clean.llm_backend = "litellm"
            config.clean.llm_model = args.llm
    if args.parser:
        config.parse.parser = args.parser
    if args.voice:
        config.synth.voice = args.voice
    if args.format:
        config.synth.output_format = args.format

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        stem = args.pdf.stem.replace(" ", "_")
        output_dir = Path("output") / stem

    print(f"PDF:       {args.pdf}")
    print(f"Output:    {output_dir}")
    print(f"Parser:    {config.parse.parser}")
    print(f"LLM:       {config.clean.llm_backend} ({config.clean.llm_model})")
    print(f"TTS:       {config.synth.tts_engine} (voice: {config.synth.voice})")
    print(f"Format:    {config.synth.output_format}")

    # Run the pipeline
    from pdf2audiobook.pipeline import run_pipeline

    try:
        result = run_pipeline(args.pdf.resolve(), output_dir.resolve(), config)
        print(f"\nAudiobook created: {result}")
    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress has been saved — run the same command to resume.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
