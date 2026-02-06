"""AIBharath TTS Plugin for LiveKit Agents

Reference: https://docs.livekit.io/agents/plugins/tts/
"""

import logging
import uuid
from dataclasses import dataclass

import aiohttp
from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("aibharath-tts")


@dataclass
class TTSOptions:
    base_url: str = "http://localhost:8001"
    speaker: str = "Divya"
    language: str = "hindi"
    speed: str = "normal"
    sample_rate: int = 44100  # TTS server returns 44100Hz


class TTS(tts.TTS):
    """AIBharath Indic Parler TTS.

    Note: TTSCapabilities(streaming=False) means SDK uses synthesize() method.
    The _use_streaming_endpoint flag controls whether we use /synthesize/stream HTTP endpoint.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8001",
        speaker: str = "Divya",
        language: str = "hindi",
        speed: str = "normal",
        sample_rate: int = 44100,
        use_streaming_endpoint: bool = True,  # Use streaming for lower latency
    ):
        # streaming=False tells SDK to use synthesize() method, not stream()
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._use_streaming_endpoint = use_streaming_endpoint
        self._opts = TTSOptions(
            base_url=base_url,
            speaker=speaker,
            language=language,
            speed=speed,
            sample_rate=sample_rate,
        )
        self._session: aiohttp.ClientSession | None = None

    def update_options(self, *, language: str | None = None, speaker: str | None = None):
        """Update language/speaker for multilanguage support."""
        if language:
            self._opts.language = language
        if speaker:
            self._opts.speaker = speaker

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
    ) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=text, opts=self._opts, conn_options=conn_options, use_streaming_endpoint=self._use_streaming_endpoint)


class ChunkedStream(tts.ChunkedStream):
    """TTS synthesis stream - uses HTTP streaming endpoint for lower latency."""

    def __init__(self, *, tts: TTS, input_text: str, opts: TTSOptions, conn_options: APIConnectOptions, use_streaming_endpoint: bool = True):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._use_streaming_endpoint = use_streaming_endpoint

    async def _run(self, output_emitter):
        session = self._tts._ensure_session()
        text_preview = self._input_text[:50].replace('\n', ' ')

        if self._use_streaming_endpoint:
            await self._run_streaming(session, output_emitter, text_preview)
        else:
            await self._run_non_streaming(session, output_emitter, text_preview)

    async def _run_streaming(self, session, output_emitter, text_preview):
        """Stream audio chunks as they're generated."""
        logger.info(f"TTS streaming request: '{text_preview}' [{self._opts.language}]")

        async with session.post(
            f"{self._opts.base_url}/synthesize/stream",
            json={
                "text": self._input_text,
                "speaker": self._opts.speaker,
                "language": self._opts.language,
                "speed": self._opts.speed,
                "output_format": "pcm",
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"TTS streaming error {resp.status}: {error}")
                raise tts.TTSError(f"TTS error: {resp.status}")

            sample_rate = int(resp.headers.get("X-Sample-Rate", self._opts.sample_rate))
            logger.info(f"TTS initializing emitter: sample_rate={sample_rate}Hz, mime_type=audio/pcm")

            output_emitter.initialize(
                request_id=str(uuid.uuid4()),
                sample_rate=sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )

            total_bytes = 0
            chunk_count = 0
            async for chunk in resp.content.iter_chunked(8192):  # 8KB chunks
                if chunk:
                    output_emitter.push(chunk)
                    total_bytes += len(chunk)
                    chunk_count += 1
                    if chunk_count == 1:
                        logger.info(f"TTS first chunk received ({len(chunk)} bytes), starting playback")
                    elif chunk_count % 10 == 0:
                        logger.debug(f"TTS chunk {chunk_count}: {total_bytes} bytes total")

            # Flush buffered audio before returning
            output_emitter.flush()
            duration_sec = total_bytes / 2 / sample_rate  # int16 = 2 bytes
            logger.info(f"TTS streaming complete: {total_bytes} bytes ({duration_sec:.2f}s) in {chunk_count} chunks")

    async def _run_non_streaming(self, session, output_emitter, text_preview):
        """Wait for full audio before pushing."""
        logger.info(f"TTS request: '{text_preview}' [{self._opts.language}]")

        async with session.post(
            f"{self._opts.base_url}/synthesize",
            json={
                "text": self._input_text,
                "speaker": self._opts.speaker,
                "language": self._opts.language,
                "speed": self._opts.speed,
                "output_format": "pcm",
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"TTS error {resp.status}: {error}")
                raise tts.TTSError(f"TTS error: {resp.status}")

            audio_data = await resp.read()
            sample_rate = int(resp.headers.get("X-Sample-Rate", self._opts.sample_rate))

            logger.info(f"TTS response: {len(audio_data)} bytes, {sample_rate}Hz")

            output_emitter.initialize(
                request_id=str(uuid.uuid4()),
                sample_rate=sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )
            output_emitter.push(audio_data)
            output_emitter.flush()
            logger.info("TTS audio pushed to emitter")
