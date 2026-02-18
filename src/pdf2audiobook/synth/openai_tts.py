"""OpenAI TTS engine — cloud API."""

from __future__ import annotations

from pathlib import Path

from pdf2audiobook.synth.base import TTSEngine


class OpenAITTS(TTSEngine):
    """OpenAI text-to-speech API."""

    def __init__(self, voice: str = "nova", api_key: str = "", model: str = "tts-1") -> None:
        self._voice = voice
        self._api_key = api_key
        self._model = model
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def sample_rate(self) -> int:
        return 24000

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key or None)
        except ImportError:
            raise RuntimeError(
                "OpenAI SDK not installed. Install with: pip install 'pdf2audiobook[openai]'"
            )

    def synthesize(self, text: str, output_path: Path) -> Path:
        self._ensure_client()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # OpenAI returns MP3 by default; we request WAV-compatible format
        response = self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="wav",
        )
        response.write_to_file(str(output_path))
        return output_path
