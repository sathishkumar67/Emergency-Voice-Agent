"""AIBharath Indic Conformer STT Plugin for LiveKit Agents

Connects to the AIBharath STT HTTP service for transcription.
"""

import os
import io
import wave
import logging
import time
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

DEBUG_FLOW = os.getenv("DEBUG_FLOW", "false").lower() == "true"

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("aibharath-conformer-stt")

SUPPORTED_LANGUAGES = [
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai",
    "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"
]

TARGET_SAMPLE_RATE = 16000


@dataclass
class STTOptions:
    base_url: str
    language: str
    decode_type: str
    translate_to_english: bool


class STT(stt.STT):
    """AIBharath Indic Conformer STT via HTTP service."""

    def __init__(
        self,
        *,
        base_url: str = "http://192.168.1.26:8002",
        language: str = "hi",
        decode_type: str = "ctc",
        translate_to_english: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._opts = STTOptions(
            base_url=base_url,
            language=language if language in SUPPORTED_LANGUAGES else "hi",
            decode_type=decode_type,
            translate_to_english=translate_to_english,
        )
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return "indic-conformer-600m"

    @property
    def provider(self) -> str:
        return "ai4bharat"

    def update_options(
        self,
        *,
        language: str | None = None,
        translate_to_english: bool | None = None,
    ):
        """Update options dynamically for multilanguage support."""
        if language and language in SUPPORTED_LANGUAGES:
            self._opts.language = language
            logger.info(f"STT language updated to: {language}")
        if translate_to_english is not None:
            self._opts.translate_to_english = translate_to_english
            logger.info(f"STT translate_to_english updated to: {translate_to_english}")

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
        logger.info(f"STT input: {duration:.2f}s audio at {sample_rate}Hz")

        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(TARGET_SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())
        wav_buffer.seek(0)

        # Send to HTTP service
        session = self._ensure_session()
        stt_start = time.time()
        try:
            form = aiohttp.FormData()
            form.add_field('file', wav_buffer, filename='audio.wav', content_type='audio/wav')
            form.add_field('language', lang)
            form.add_field('decode_type', self._opts.decode_type)
            form.add_field('translate', str(self._opts.translate_to_english).lower())

            async with session.post(
                f"{self._opts.base_url}/transcribe",
                data=form,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    original_text = data.get("text", "")
                    translated_text = data.get("translated_text")
                    stt_end = time.time()

                    # Use translated text only if translation is enabled and available
                    if self._opts.translate_to_english and translated_text:
                        text = translated_text
                        logger.info(f"[TIMING] STT+{lang}→en: {(stt_end - stt_start)*1000:.0f}ms (audio={duration:.2f}s)")
                        if DEBUG_FLOW:
                            logger.info(f"[FLOW:STT_OUT] Original [{lang}]: '{original_text}'")
                            logger.info(f"[FLOW:INDIC_TRANS] {lang}→en: '{translated_text}'")
                    else:
                        text = original_text
                        logger.info(f"[TIMING] STT ({lang}, no translation): {(stt_end - stt_start)*1000:.0f}ms (audio={duration:.2f}s)")
                        if DEBUG_FLOW:
                            logger.info(f"[FLOW:STT_OUT] [{lang}]: '{original_text}'")
                else:
                    error = await resp.text()
                    logger.error(f"STT error {resp.status}: {error}")
                    text = ""

        except Exception as e:
            logger.error(f"STT request failed: {e}")
            text = ""

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[stt.SpeechData(language=lang, text=text)],
        )
