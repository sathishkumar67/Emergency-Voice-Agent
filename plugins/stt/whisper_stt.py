"""Whisper STT Plugin for LiveKit Agents

Connects to WhisperLiveKit HTTP service for transcription.
Use with StreamAdapter and VAD for streaming support.

Usage:
    from whisper_stt import STT as WhisperSTT
    from livekit.agents import stt
    from livekit.plugins import silero

    base_stt = WhisperSTT(base_url="http://localhost:8003", language="auto")
    vad = silero.VAD.load()
    streaming_stt = stt.StreamAdapter(base_stt, vad.stream())
"""

import io
import wave
import logging
from dataclasses import dataclass

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("whisper-stt")

TARGET_SAMPLE_RATE = 16000


@dataclass
class STTOptions:
    base_url: str
    language: str


class STT(stt.STT):
    """Whisper STT via WhisperLiveKit HTTP service."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8003",
        language: str = "auto",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._opts = STTOptions(base_url=base_url, language=language)
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return "whisper"

    @property
    def provider(self) -> str:
        return "whisperlivekit"

    def update_options(self, *, language: str | None = None):
        """Update language dynamically."""
        if language:
            self._opts.language = language
            logger.info(f"Whisper STT language updated to: {language}")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if not isinstance(language, type(NOT_GIVEN)) else self._opts.language

        frames = buffer if isinstance(buffer, list) else [buffer]
        audio_chunks = []
        sample_rate = TARGET_SAMPLE_RATE

        for frame in frames:
            if isinstance(frame, rtc.AudioFrame):
                sample_rate = frame.sample_rate
                pcm = np.frombuffer(bytes(frame.data), dtype=np.int16)
                if frame.num_channels == 2:
                    pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
                audio_chunks.append(pcm)

        if not audio_chunks:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=utils.shortuuid(),
                alternatives=[stt.SpeechData(language=lang, text="")],
            )

        audio_data = np.concatenate(audio_chunks)

        # Resample to 16kHz if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = rtc.AudioResampler(
                input_rate=sample_rate,
                output_rate=TARGET_SAMPLE_RATE,
                num_channels=1
            )
            frame = rtc.AudioFrame(
                data=audio_data.tobytes(),
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(audio_data)
            )
            resampled = resampler.push(frame) + resampler.flush()
            if resampled:
                audio_data = np.frombuffer(b''.join(bytes(f.data) for f in resampled), dtype=np.int16)
            sample_rate = TARGET_SAMPLE_RATE

        duration = len(audio_data) / sample_rate
        logger.debug(f"Whisper STT input: {duration:.2f}s audio")

        # Create WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(TARGET_SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())
        wav_buffer.seek(0)

        # Send to HTTP service
        session = self._ensure_session()
        detected_lang = lang  # Default to input, updated on success
        try:
            form = aiohttp.FormData()
            form.add_field('file', wav_buffer, filename='audio.wav', content_type='audio/wav')
            form.add_field('language', lang)
            form.add_field('task', 'translate')

            async with session.post(
                f"{self._opts.base_url}/transcribe",
                data=form,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data.get("text", "")
                    detected_lang = data.get("language", lang)
                    logger.info(f"Whisper [{detected_lang}]: {text[:100]}")
                else:
                    error = await resp.text()
                    logger.error(f"Whisper STT error {resp.status}: {error}")
                    text = ""

        except Exception as e:
            logger.error(f"Whisper STT request failed: {e}")
            text = ""

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[stt.SpeechData(language=detected_lang, text=text)],
        )
