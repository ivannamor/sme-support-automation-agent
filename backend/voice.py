"""
Voice utilities: Speech-to-Text (Whisper) and Text-to-Speech (OpenAI TTS).
"""

import io
import logging
import re

from backend.secrets_provider import get_secret

logger = logging.getLogger("Voice")

TTS_VOICES = ["nova", "alloy", "echo", "fable", "onyx", "shimmer"]
TTS_DEFAULT_VOICE = "nova"


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribes audio using OpenAI Whisper.
    Returns the transcribed text string.
    Raises on API error.
    """
    from openai import OpenAI
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    text = response.text.strip()
    logger.info(f"WHISPER_TRANSCRIPTION: '{text[:80]}'")
    return text


def text_to_speech(text: str, voice: str = TTS_DEFAULT_VOICE) -> bytes:
    """
    Converts text to MP3 audio using OpenAI TTS.
    Returns raw MP3 bytes.
    Raises on API error.
    """
    from openai import OpenAI
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

    clean = _clean_for_tts(text)
    if not clean:
        return b""

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice if voice in TTS_VOICES else TTS_DEFAULT_VOICE,
        input=clean[:4096],  # API limit
    )
    logger.info(f"TTS_GENERATED: {len(clean)} chars, voice={voice}")
    return response.content


def _clean_for_tts(text: str) -> str:
    """Strips markdown and delimiters so TTS reads naturally."""
    # Remove SOURCE_DATA_SPLIT citations
    if "SOURCE_DATA_SPLIT" in text:
        text = text.split("SOURCE_DATA_SPLIT")[0]
    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers, keep content
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    # Replace code blocks with a short placeholder
    text = re.sub(r"```[\s\S]*?```", "[codice]", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Collapse extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
