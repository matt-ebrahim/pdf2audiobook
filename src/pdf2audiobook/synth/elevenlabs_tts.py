"""ElevenLabs TTS engine — cloud API, premium quality."""

from __future__ import annotations

from pathlib import Path

from pdf2audiobook.synth.base import TTSEngine


class ElevenLabsTTS(TTSEngine):
    """ElevenLabs text-to-speech API."""

    def __init__(self, voice: str = "Rachel", api_key: str = "") -> None:
        self._voice = voice
        self._api_key = api_key
        self._client = None

    @property
    def name(self) -> str:
        return "elevenlabs"

    @property
    def sample_rate(self) -> int:
        return 44100

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from elevenlabs import ElevenLabs
            self._client = ElevenLabs(api_key=self._api_key or None)
        except ImportError:
            raise RuntimeError(
                "ElevenLabs SDK not installed. Install with: pip install 'pdf2audiobook[elevenlabs]'"
            )

    def synthesize(self, text: str, output_path: Path) -> Path:
        self._ensure_client()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_iter = self._client.text_to_speech.convert(
            voice_id=self._voice,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        with open(output_path, "wb") as f:
            for chunk in audio_iter:
                f.write(chunk)

        return output_path
