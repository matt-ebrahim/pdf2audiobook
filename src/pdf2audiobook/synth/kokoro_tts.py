"""Kokoro TTS engine — local, CPU-only, fast, Apache 2.0."""

from __future__ import annotations

from pathlib import Path

from pdf2audiobook.synth.base import TTSEngine


class KokoroTTS(TTSEngine):
    """Kokoro-82M text-to-speech engine."""

    def __init__(self, voice: str = "af_heart") -> None:
        self._voice = voice
        self._pipeline = None

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def sample_rate(self) -> int:
        return 24000

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code=self._voice[0])
        except ImportError:
            raise RuntimeError(
                "Kokoro not installed. Install with: pip install kokoro soundfile"
            )

    def synthesize(self, text: str, output_path: Path) -> Path:
        self._ensure_loaded()
        import soundfile as sf

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate audio using Kokoro pipeline
        all_audio = []
        for result in self._pipeline(text, voice=self._voice):
            if result.audio is not None:
                all_audio.append(result.audio)

        if not all_audio:
            raise RuntimeError(f"Kokoro produced no audio for text: {text[:80]}...")

        import numpy as np
        combined = np.concatenate(all_audio)
        sf.write(str(output_path), combined, self.sample_rate)
        return output_path
