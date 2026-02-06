"""Indic TTS Plugin for LiveKit Agents

Connects to the Indic TTS HTTP API for Indian language synthesis.
API Server: http://192.168.1.26:8004

Supported Languages: English (en), Hindi (hi), Kannada (kn), Malayalam (ml),
                    Marathi (mr), Tamil (ta), Telugu (te)

Reference: https://github.com/AI4Bharat/Indic-TTS
"""

import os
import logging
import uuid
import io
import wave
from dataclasses import dataclass
from typing import Optional

import aiohttp
from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("indic-tts")

# API Configuration
INDIC_TTS_URL = os.getenv("INDIC_TTS_URL", "http://192.168.1.26:8004")

# Supported languages (all 7)
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "speakers": ["female", "male"], "default": "female"},
    "hi": {"name": "Hindi", "speakers": ["female", "male"], "default": "female"},
    "kn": {"name": "Kannada", "speakers": ["female", "male"], "default": "female"},
    "ml": {"name": "Malayalam", "speakers": ["female", "male"], "default": "female"},
    "mr": {"name": "Marathi", "speakers": ["female", "male"], "default": "female"},
    "ta": {"name": "Tamil", "speakers": ["female", "male"], "default": "female"},
    "te": {"name": "Telugu", "speakers": ["female", "male"], "default": "female"},
}

# Direct language mapping (all supported natively)
LANG_ROUTING = {
    "en": "en",
    "hi": "hi",
    "kn": "kn",
    "ml": "ml",
    "mr": "mr",
    "ta": "ta",
    "te": "te",
}


def get_routed_language(lang_code: str) -> Optional[str]:
    return LANG_ROUTING.get(lang_code)


def is_language_supported(lang_code: str) -> bool:
    return lang_code in LANG_ROUTING


@dataclass
class IndicTTSOptions:
    base_url: str = INDIC_TTS_URL
    language: str = "kn"
    speaker: str = "female"


class TTS(tts.TTS):
    """Indic TTS via HTTP API."""

    def __init__(
        self,
        *,
        base_url: str = INDIC_TTS_URL,
        language: str = "kn",
        speaker: str = "female",
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=22050,
            num_channels=1,
        )

        routed_lang = get_routed_language(language)
        if not routed_lang:
            logger.warning(f"Language '{language}' not supported, defaulting to Kannada")
            routed_lang = "kn"

        self._opts = IndicTTSOptions(
            base_url=base_url,
            language=routed_lang,
            speaker=speaker,
        )
        self._session: Optional[aiohttp.ClientSession] = None

    def update_options(
        self,
        *,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
    ):
        if language:
            routed_lang = get_routed_language(language)
            if routed_lang:
                self._opts.language = routed_lang
                logger.info(f"Indic TTS language updated to: {routed_lang}")
            else:
                logger.warning(f"Language '{language}' not supported")

        if speaker and speaker in ["female", "male"]:
            self._opts.speaker = speaker
            logger.info(f"Indic TTS speaker updated to: {speaker}")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "IndicTTSStream":
        return IndicTTSStream(
            tts=self, input_text=text, opts=self._opts, conn_options=conn_options
        )


class IndicTTSStream(tts.ChunkedStream):
    """Synthesizes audio via Indic TTS HTTP API."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: IndicTTSOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._tts = tts

    async def _run(self, output_emitter):
        text_preview = self._input_text[:50].replace('\n', ' ')
        logger.info(f"Indic TTS: '{text_preview}...' [lang={self._opts.language}, speaker={self._opts.speaker}]")

        session = self._tts._ensure_session()

        payload = {
            "text": self._input_text,
            "language": self._opts.language,
            "speaker": self._opts.speaker,
            "output_format": "wav",
        }

        try:
            async with session.post(
                f"{self._opts.base_url}/synthesize",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"TTS API error {resp.status}: {error}")

                wav_data = await resp.read()

                # Parse WAV to get PCM data
                with io.BytesIO(wav_data) as wav_buffer:
                    with wave.open(wav_buffer, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        pcm_data = wav_file.readframes(wav_file.getnframes())

                output_emitter.initialize(
                    request_id=str(uuid.uuid4()),
                    sample_rate=sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )
                output_emitter.push(pcm_data)

        except Exception as e:
            logger.error(f"Indic TTS synthesis failed: {e}")
            raise


def get_tts_for_lang(lang_code: str) -> Optional[TTS]:
    if not is_language_supported(lang_code):
        return None
    return TTS(language=lang_code)
