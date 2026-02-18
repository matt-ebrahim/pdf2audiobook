"""Base class for TTS engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TTSEngine(ABC):
    """Abstract base for text-to-speech engines."""

    @abstractmethod
    def synthesize(self, text: str, output_path: Path) -> Path:
        """Convert text to audio and save to output_path (WAV format).

        Returns the path to the saved audio file.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def sample_rate(self) -> int:
        return 24000
