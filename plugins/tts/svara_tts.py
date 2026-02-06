"""Svara TTS Plugin for LiveKit Agents

Open-source Indic TTS using Svara-TTS-v1 (3B params, 19 languages).
Supports Hindi, Kannada, Tamil, Telugu, Bengali, and more Indian languages.

Reference: https://github.com/Kenpath/svara-tts-inference
Model: kenpath/svara-tts-v1
"""

import logging
import uuid
import time
from dataclasses import dataclass

import aiohttp
from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("svara-tts")

SAMPLE_RATE = 24000

# Language code to voice name mapping
# Format: {lang_code}_{gender} -> "Language (Gender)"
SVARA_VOICE_MAP = {
    "hi_male": "Hindi (Male)",
    "hi_female": "Hindi (Female)",
    "kn_male": "Kannada (Male)",
    "kn_female": "Kannada (Female)",
    "ta_male": "Tamil (Male)",
    "ta_female": "Tamil (Female)",
    "te_male": "Telugu (Male)",
    "te_female": "Telugu (Female)",
    "bn_male": "Bengali (Male)",
    "bn_female": "Bengali (Female)",
    "mr_male": "Marathi (Male)",
    "mr_female": "Marathi (Female)",
    "gu_male": "Gujarati (Male)",
    "gu_female": "Gujarati (Female)",
    "ml_male": "Malayalam (Male)",
    "ml_female": "Malayalam (Female)",
    "pa_male": "Punjabi (Male)",
    "pa_female": "Punjabi (Female)",
    "en_male": "English (Male)",
    "en_female": "English (Female)",
    "as_male": "Assamese (Male)",
    "as_female": "Assamese (Female)",
    "ne_male": "Nepali (Male)",
    "ne_female": "Nepali (Female)",
    "sa_male": "Sanskrit (Male)",
    "sa_female": "Sanskrit (Female)",
    "or_male": "Odia (Male)",
    "or_female": "Odia (Female)",
}

# Reverse mapping for convenience
LANG_GENDER_TO_VOICE_ID = {v: k for k, v in SVARA_VOICE_MAP.items()}


@dataclass
class SvaraOptions:
    base_url: str = "http://localhost:8890"
    voice: str = "kn_male"  # Default to Kannada male
    response_format: str = "pcm"


class TTS(tts.TTS):
    """Svara TTS via REST API."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8890",
        voice: str = "kn_male",
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._opts = SvaraOptions(base_url=base_url, voice=voice)
        self._session: aiohttp.ClientSession | None = None

    def update_options(self, *, voice: str | None = None):
        """Update voice dynamically.

        Args:
            voice: Voice ID in format "{lang}_{gender}" (e.g., "kn_male", "hi_female")
        """
        if voice:
            self._opts.voice = voice
            logger.info(f"Svara TTS voice updated to: {voice}")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SvaraStream":
        return SvaraStream(
            tts=self, input_text=text, opts=self._opts, conn_options=conn_options
        )


class SvaraStream(tts.ChunkedStream):
    """Streams audio from Svara TTS endpoint."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: SvaraOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter):
        session = self._tts._ensure_session()
        text_preview = self._input_text[:50].replace('\n', ' ')
        text_len = len(self._input_text)

        # Get the voice name from voice ID
        voice_name = SVARA_VOICE_MAP.get(self._opts.voice, self._opts.voice)
        logger.info(f"Svara TTS request: '{text_preview}' [voice={self._opts.voice} -> {voice_name}]")

        tts_start = time.time()
        async with session.post(
            f"{self._opts.base_url}/v1/text-to-speech",
            data={
                "text": self._input_text,
                "voice": voice_name,
                "model_id": "svara-tts-v1",
                "stream": "true",
                "response_format": "pcm",
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"Svara TTS error {resp.status}: {error}")
                raise tts.TTSError(f"Svara TTS error: {resp.status}")

            output_emitter.initialize(
                request_id=str(uuid.uuid4()),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                mime_type="audio/pcm",
            )

            total_bytes = 0
            chunk_count = 0
            first_chunk_time = None
            async for chunk in resp.content.iter_chunked(4096):
                if chunk:
                    output_emitter.push(chunk)
                    total_bytes += len(chunk)
                    chunk_count += 1
                    if chunk_count == 1:
                        first_chunk_time = time.time()
                        logger.info(f"[TIMING] TTS first chunk: {(first_chunk_time - tts_start)*1000:.0f}ms")

            output_emitter.flush()
            tts_end = time.time()
            duration = total_bytes / 2 / SAMPLE_RATE
            logger.info(f"[TIMING] TTS complete: {(tts_end - tts_start)*1000:.0f}ms ({text_len} chars → {duration:.2f}s audio)")


# Helper function to build voice ID from language and gender
def get_voice_id(lang_code: str, gender: str = "male") -> str:
    """Build voice ID from language code and gender.

    Args:
        lang_code: ISO language code (hi, kn, ta, etc.)
        gender: "male" or "female"

    Returns:
        Voice ID like "kn_male"
    """
    voice_id = f"{lang_code}_{gender}"
    if voice_id not in SVARA_VOICE_MAP:
        logger.warning(f"Unknown voice ID: {voice_id}, falling back to kn_male")
        return "kn_male"
    return voice_id
