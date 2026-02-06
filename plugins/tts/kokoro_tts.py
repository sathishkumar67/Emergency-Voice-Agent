"""Kokoro TTS Plugin for LiveKit Agents

OpenAI-compatible TTS using Kokoro-82M FastAPI server.
Supports Hindi (hf_alpha, hf_beta, hm_omega, hm_psi) and English voices.

Reference: https://github.com/remsky/Kokoro-FastAPI
"""

import logging
import uuid
from dataclasses import dataclass

import aiohttp
from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("kokoro-tts")

SAMPLE_RATE = 24000


@dataclass
class KokoroOptions:
    base_url: str = "http://localhost:8880"
    voice: str = "hm_omega"
    speed: float = 1.0


class TTS(tts.TTS):
    """Kokoro TTS via OpenAI-compatible API."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8880",
        voice: str = "hm_omega",
        speed: float = 1.0,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._opts = KokoroOptions(base_url=base_url, voice=voice, speed=speed)
        self._session: aiohttp.ClientSession | None = None

    def update_options(self, *, voice: str | None = None, speed: float | None = None):
        """Update voice/speed dynamically."""
        if voice:
            self._opts.voice = voice
        if speed:
            self._opts.speed = speed

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
    ) -> "KokoroStream":
        return KokoroStream(
            tts=self, input_text=text, opts=self._opts, conn_options=conn_options
        )


class KokoroStream(tts.ChunkedStream):
    """Streams audio from Kokoro OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: KokoroOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter):
        session = self._tts._ensure_session()
        text_preview = self._input_text[:50].replace('\n', ' ')
        logger.info(f"Kokoro TTS request: '{text_preview}' [voice={self._opts.voice}]")

        async with session.post(
            f"{self._opts.base_url}/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": self._input_text,
                "voice": self._opts.voice,
                "response_format": "pcm",
                "speed": self._opts.speed,
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"Kokoro TTS error {resp.status}: {error}")
                raise tts.TTSError(f"Kokoro TTS error: {resp.status}")

            output_emitter.initialize(
                request_id=str(uuid.uuid4()),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                mime_type="audio/pcm",
            )

            total_bytes = 0
            chunk_count = 0
            async for chunk in resp.content.iter_chunked(4096):
                if chunk:
                    output_emitter.push(chunk)
                    total_bytes += len(chunk)
                    chunk_count += 1
                    if chunk_count == 1:
                        logger.info(f"Kokoro first chunk ({len(chunk)} bytes)")

            output_emitter.flush()
            duration = total_bytes / 2 / SAMPLE_RATE
            logger.info(f"Kokoro TTS complete: {total_bytes} bytes ({duration:.2f}s)")
