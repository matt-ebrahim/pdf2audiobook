"""Chatterbox TTS engine — local, GPU required, best local quality, MIT license."""

from __future__ import annotations

from pathlib import Path

from pdf2audiobook.synth.base import TTSEngine


class ChatterboxTTS(TTSEngine):
    """Chatterbox text-to-speech engine by Resemble AI."""

    def __init__(self, voice: str = "") -> None:
        self._voice_path = voice if voice else None
        self._model = None

    @property
    def name(self) -> str:
        return "chatterbox"

    @property
    def sample_rate(self) -> int:
        return 24000

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from chatterbox.tts import ChatterboxTTS as CBModel
            self._model = CBModel.from_pretrained(device="cuda")
        except ImportError:
            raise RuntimeError(
                "Chatterbox not installed. Install with: pip install chatterbox-tts"
            )

    def synthesize(self, text: str, output_path: Path) -> Path:
        self._ensure_loaded()
        import torchaudio

        output_path.parent.mkdir(parents=True, exist_ok=True)

        wav = self._model.generate(
            text,
            audio_prompt_path=self._voice_path,
        )
        torchaudio.save(str(output_path), wav, self.sample_rate)
        return output_path
