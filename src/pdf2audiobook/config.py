"""Configuration loading from TOML files with sensible defaults."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParseConfig:
    parser: str = "auto"
    sample_pages: int = 5


@dataclass
class CleanConfig:
    # LiteLLM model string — any model supported by litellm.completion()
    # Examples: "gpt-4o-mini", "claude-sonnet-4-20250514", "gemini/gemini-2.0-flash"
    llm_model: str = "gpt-4o-mini"
    # Set to "none" to use regex-only fallback (no LLM)
    llm_backend: str = "litellm"
    skip_figures: bool = True
    skip_tables: bool = True
    skip_references: bool = True


@dataclass
class ChunkConfig:
    max_chars: int = 400
    splitter: str = "spacy"
    spacy_model: str = "en_core_web_sm"


@dataclass
class SynthConfig:
    tts_engine: str = "kokoro"
    voice: str = "af_heart"
    output_format: str = "m4b"
    pause_sentence_ms: int = 300
    pause_paragraph_ms: int = 700
    pause_chapter_ms: int = 2000
    target_lufs: int = -16
    parallel_workers: int = 0  # 0 = auto-detect


@dataclass
class ApiKeys:
    openai: str = ""
    anthropic: str = ""
    elevenlabs: str = ""


@dataclass
class Config:
    parse: ParseConfig = field(default_factory=ParseConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    api_keys: ApiKeys = field(default_factory=ApiKeys)


def load_config(path: Path | None = None) -> Config:
    """Load config from a TOML file, falling back to defaults."""
    cfg = Config()
    if path is None or not path.exists():
        return cfg

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    if "parse" in raw:
        for k, v in raw["parse"].items():
            if hasattr(cfg.parse, k):
                setattr(cfg.parse, k, v)
    if "clean" in raw:
        for k, v in raw["clean"].items():
            if hasattr(cfg.clean, k):
                setattr(cfg.clean, k, v)
    if "chunk" in raw:
        for k, v in raw["chunk"].items():
            if hasattr(cfg.chunk, k):
                setattr(cfg.chunk, k, v)
    if "synth" in raw:
        for k, v in raw["synth"].items():
            if hasattr(cfg.synth, k):
                setattr(cfg.synth, k, v)
    if "api_keys" in raw:
        for k, v in raw["api_keys"].items():
            if hasattr(cfg.api_keys, k):
                setattr(cfg.api_keys, k, v)

    return cfg
